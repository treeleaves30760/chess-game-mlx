// SPDX-License-Identifier: MIT
// engine/include/search/multi_ponder.hpp
//
// MultiPonderManager<Traits> — manages 5 parallel ponder trees.
//
// While the opponent is thinking, the engine pre-computes our best response
// for the top-k predicted opponent moves, sharing a single Batcher (and
// therefore a single GPU dispatch queue) across all trees.
//
// Design points:
//  * Shared Batcher: all five MCTS instances submit leaf-eval requests to the
//    same owning Batcher owned by this manager.  Cross-tree batching is thus
//    automatic — the GPU sees one big batch regardless of which tree the leaf
//    came from.
//  * Budget weights: each tree gets a fraction of the total CPU threads
//    (proportional to the weight); trees with larger weights run more threads
//    and accumulate more nodes per second.
//  * Progress emitter: a dedicated thread fires the on_event("ponder_progress")
//    callback every ~500 ms.
//  * Lifecycle: start() → {opponent_played() | stop()} → optional take_promoted_mcts()
//  * Thread safety: public methods are NOT re-entrant, but stop() is safe to
//    call from any thread concurrently with ongoing ponder work.

#pragma once

#include "nn/backend.hpp"
#include "nn/batcher.hpp"
#include "search/mcts.hpp"
#include "search/policy_utils.hpp"

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// nlohmann/json for event payloads.
#include <nlohmann_json/json.hpp>

namespace chess_mlx::search {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// PonderTree — per-hypothesis state
// ---------------------------------------------------------------------------
template <typename Traits>
struct PonderTree {
    int  tree_id{-1};
    typename Traits::Move opp_move{};  // the opponent move this tree assumes
    float opp_prob{0.0f};              // NN prior probability for this move
    double budget_weight{0.0};         // fraction of threads allocated

    std::unique_ptr<search::MCTS<Traits>> mcts;
    std::thread  worker;

    std::atomic<bool>     running{false};
    std::atomic<uint64_t> nodes{0};
    std::atomic<int>      seldepth{0};
    std::atomic<float>    value{0.0f};  // current root value (stm-relative)

    // Snapshot of top PV (string form, e.g. "Nf3", "Nc6").
    // Protected by pv_mtx.
    std::vector<std::string> top_pv;
    mutable std::mutex       pv_mtx;

    // Not copyable / movable (atomics + mutex).
    PonderTree() = default;
    PonderTree(const PonderTree&) = delete;
    PonderTree& operator=(const PonderTree&) = delete;

    ~PonderTree() {
        if (mcts) mcts->cancel();
        if (worker.joinable()) worker.join();
    }
};

// ---------------------------------------------------------------------------
// MultiPonderManager<Traits>
// ---------------------------------------------------------------------------
template <typename Traits>
class MultiPonderManager {
public:
    using Position = typename Traits::Position;
    using Move     = typename Traits::Move;
    using MCTSType = search::MCTS<Traits>;

    // `backend`         — shared NN backend (used by the shared Batcher).
    // `threads_per_tree`— CPU threads per MCTS tree (budget-scaled below).
    // `input_size`      — flat NN input tensor size (e.g. 64*19 for chess).
    // `policy_size`     — NN output policy size (e.g. 4672 for chess).
    explicit MultiPonderManager(std::shared_ptr<nn::NNBackend> backend,
                                int threads_per_tree = 1,
                                std::size_t input_size  = 64 * 19,
                                std::size_t policy_size = 4672)
        : backend_(std::move(backend))
        , threads_per_tree_(threads_per_tree)
        , input_size_(input_size)
        , policy_size_(policy_size)
        // Owning Batcher — shared across all trees.
        // With 5 trees each running multiple threads, a 2ms timeout ensures
        // cross-tree batching without excessive latency.
        , shared_batcher_(backend_,
              nn::BatcherConfig{
                  /*max_batch_size=*/256,
                  /*flush_timeout_us=*/500})
    {
        for (int i = 0; i < kMaxTrees; ++i) trees_[i].tree_id = i;
    }

    ~MultiPonderManager() { stop(); }

    // -----------------------------------------------------------------------
    // start() — begin multi-ponder from `root`.
    //
    // root      : current position (it is the opponent's turn to move).
    // k         : how many top opponent moves to predict (max kMaxTrees=5).
    // weights   : budget fraction for each tree (must sum to ~1, size >= k).
    // on_event  : callback invoked for policy_preview, ponder_progress,
    //             ponder_hit, ponder_miss notifications (may be called from
    //             background threads — must be thread-safe).
    //
    // Emits policy_preview via on_event before returning.
    // Returns immediately; ponder work runs in background threads.
    // -----------------------------------------------------------------------
    void start(const Position& root,
               int k,
               const std::vector<double>& weights,
               std::function<void(json)> on_event) {
        stop();  // cancel any previous ponder first

        on_event_   = std::move(on_event);
        ponder_stopped_.store(false, std::memory_order_release);
        promoted_tree_idx_.store(-1, std::memory_order_relaxed);

        // 1. Compute top-k policy moves (single NN forward).
        const int actual_k = std::min(k, kMaxTrees);
        auto top_moves = search::top_k_policy_moves<Traits>(
            *backend_, root, input_size_, actual_k);

        // If the NN gave fewer moves than k (terminal or near-terminal), clamp.
        const int n = static_cast<int>(top_moves.size());

        // Emit policy_preview.
        {
            json preview_moves = json::array();
            for (auto& [mv, prob] : top_moves) {
                json entry;
                entry["uci"]  = Traits::move_to_string(mv);
                entry["prob"] = static_cast<double>(prob);
                preview_moves.push_back(entry);
            }
            json notif;
            notif["jsonrpc"] = "2.0";
            notif["method"]  = "policy_preview";
            notif["params"]["moves"] = preview_moves;
            if (on_event_) on_event_(notif);
        }

        if (n == 0) return;  // no moves to ponder

        // 2. Build one MCTS tree per hypothesis.
        active_trees_ = n;
        for (int i = 0; i < n; ++i) {
            auto& t = trees_[static_cast<std::size_t>(i)];
            t.opp_move     = top_moves[static_cast<std::size_t>(i)].first;
            t.opp_prob     = top_moves[static_cast<std::size_t>(i)].second;
            t.budget_weight = (static_cast<int>(weights.size()) > i)
                              ? weights[static_cast<std::size_t>(i)]
                              : 1.0 / static_cast<double>(n);

            // Threads allocated: at least 1, proportional to weight.
            const int tree_threads = std::max(1,
                static_cast<int>(std::round(
                    t.budget_weight * static_cast<double>(threads_per_tree_ * n))));

            // Build position after hypothetical opp move.
            Position child_pos = root;
            Traits::apply(child_pos, t.opp_move);
            opp_positions_[static_cast<std::size_t>(i)] = child_pos;

            // Build MCTS with shared batcher.
            typename MCTSType::Config cfg;
            cfg.threads     = tree_threads;
            cfg.input_size  = input_size_;
            cfg.policy_size = policy_size_;
            t.mcts = std::make_unique<MCTSType>(
                backend_, cfg, &shared_batcher_);

            t.nodes.store(0, std::memory_order_relaxed);
            t.seldepth.store(0, std::memory_order_relaxed);
            t.value.store(0.0f, std::memory_order_relaxed);
        }

        // Nullify unused trees.
        for (int i = n; i < kMaxTrees; ++i) {
            trees_[static_cast<std::size_t>(i)].mcts.reset();
            trees_[static_cast<std::size_t>(i)].running.store(false);
        }

        // 3. Launch per-tree worker threads.
        for (int i = 0; i < n; ++i) {
            auto& t = trees_[static_cast<std::size_t>(i)];
            t.running.store(true, std::memory_order_release);
            const Position child_pos = opp_positions_[static_cast<std::size_t>(i)];

            t.worker = std::thread([this, i, child_pos]() {
                auto& tree = trees_[static_cast<std::size_t>(i)];
                search::TimeControl tc;
                tc.infinite = true;  // run until stop() is called
                tree.mcts->set_progress_callback(
                    [this, i, &tree, child_pos](
                        const SearchResult<Traits>& r) {
                        // Update snapshot atomics.
                        tree.nodes.store(r.nodes, std::memory_order_relaxed);
                        tree.seldepth.store(r.seldepth, std::memory_order_relaxed);
                        if (!r.top_values.empty()) {
                            tree.value.store(r.top_values[0],
                                std::memory_order_relaxed);
                        }
                        // Update PV string snapshot.
                        if (!r.top_pvs.empty()) {
                            std::lock_guard<std::mutex> lk(tree.pv_mtx);
                            tree.top_pv.clear();
                            for (const auto& mv : r.top_pvs[0]) {
                                tree.top_pv.push_back(
                                    Traits::move_to_string(mv));
                            }
                        }
                    });
                tree.mcts->search(child_pos, tc);
                tree.running.store(false, std::memory_order_release);
            });
        }

        // 4. Start progress emitter thread.
        progress_thread_ = std::thread([this, n]() {
            progress_emitter_loop(n);
        });
    }

    // -----------------------------------------------------------------------
    // opponent_played() — called when the opponent actually makes a move.
    //
    // Returns: the tree index (0..k-1) if the move was in our top-k, -1 otherwise.
    //
    // Side effects:
    //   Hit:  all other trees cancelled; winning tree's current best move is
    //         reported via on_event("ponder_hit", ...); the winning MCTS is
    //         promoted (promote_root) so it can continue searching our response.
    //   Miss: all trees cancelled; on_event("ponder_miss", ...) emitted.
    // -----------------------------------------------------------------------
    int opponent_played(const Move& actual_move) {
        const int n = active_trees_;
        int hit_idx = -1;

        // Find matching tree by comparing policy indices.
        for (int i = 0; i < n; ++i) {
            if (move_equals(trees_[static_cast<std::size_t>(i)].opp_move,
                            actual_move)) {
                hit_idx = i;
                break;
            }
        }

        if (hit_idx >= 0) {
            // Stop ALL trees (including the winner) so we can safely snapshot.
            for (int i = 0; i < n; ++i) {
                auto& t = trees_[static_cast<std::size_t>(i)];
                if (t.mcts) t.mcts->stop();
            }
            // Stop the progress emitter.
            {
                std::lock_guard<std::mutex> lk(progress_mtx_);
                ponder_stopped_.store(true, std::memory_order_release);
            }
            progress_cv_.notify_all();
            if (progress_thread_.joinable()) progress_thread_.join();

            // Join ALL workers (including the winner).
            for (int i = 0; i < n; ++i) {
                auto& t = trees_[static_cast<std::size_t>(i)];
                if (t.worker.joinable()) t.worker.join();
            }

            // Promote the hit tree.
            auto& winner = trees_[static_cast<std::size_t>(hit_idx)];
            promoted_tree_idx_.store(hit_idx, std::memory_order_relaxed);

            // Build the "after our ponder root" position.
            // Safe to snapshot now — winner's worker is joined.
            const Position& hit_pos =
                opp_positions_[static_cast<std::size_t>(hit_idx)];

            auto snap = winner.mcts->snapshot(hit_pos);
            const std::string best_uci =
                Traits::is_null_move(snap.best_move)
                ? ""
                : Traits::move_to_string(snap.best_move);
            const int score_cp = snap.top_values.empty()
                ? 0
                : value_to_cp_internal(snap.top_values[0]);

            // Emit ponder_hit.
            {
                json notif;
                notif["jsonrpc"] = "2.0";
                notif["method"]  = "ponder_hit";
                notif["params"]["tree"]            = hit_idx;
                notif["params"]["instant_bestmove"] = best_uci;
                notif["params"]["score_cp"]        = score_cp;
                if (on_event_) on_event_(notif);
            }

            // Clean up non-winning trees.
            for (int i = 0; i < n; ++i) {
                if (i != hit_idx) {
                    trees_[static_cast<std::size_t>(i)].mcts.reset();
                }
            }

            // Transfer promoted MCTS.  Reset cancel so the promoted MCTS
            // can be used for a fresh search immediately.
            winner.mcts->reset_cancel();
            promoted_mcts_ = std::move(winner.mcts);
            promoted_position_ = hit_pos;
        } else {
            // Miss — stop everything.
            stop_all_trees(n);
            {
                std::lock_guard<std::mutex> lk(progress_mtx_);
                ponder_stopped_.store(true, std::memory_order_release);
            }
            progress_cv_.notify_all();
            if (progress_thread_.joinable()) progress_thread_.join();

            json notif;
            notif["jsonrpc"] = "2.0";
            notif["method"]  = "ponder_miss";
            notif["params"]["trees_discarded"] = n;
            if (on_event_) on_event_(notif);
        }

        active_trees_ = 0;
        return hit_idx;
    }

    // -----------------------------------------------------------------------
    // stop() — cancel all running trees, join all threads.
    // Safe to call even if no ponder is active.
    // -----------------------------------------------------------------------
    void stop() {
        const int n = active_trees_;
        stop_all_trees(n);
        {
            std::lock_guard<std::mutex> lk(progress_mtx_);
            ponder_stopped_.store(true, std::memory_order_release);
        }
        progress_cv_.notify_all();
        if (progress_thread_.joinable()) progress_thread_.join();
        active_trees_ = 0;
    }

    // -----------------------------------------------------------------------
    // take_promoted_mcts() — if a ponder_hit occurred, the caller can
    // extract the promoted MCTS (already pointing at the post-opp-move
    // position) and continue searching from it.  Returns nullptr if no hit.
    // After this call the internal pointer is cleared.
    // -----------------------------------------------------------------------
    std::unique_ptr<MCTSType> take_promoted_mcts() {
        return std::move(promoted_mcts_);
    }

    // Position after the opponent's move (valid only after a hit).
    const Position& promoted_position() const { return promoted_position_; }

    // Tree count currently active.
    int active_tree_count() const { return active_trees_; }

    // Access tree stats (for testing / reporting).
    const PonderTree<Traits>& tree(int i) const {
        return trees_[static_cast<std::size_t>(i)];
    }

private:
    // Maximum number of parallel ponder trees.
    static constexpr int kMaxTrees = 5;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Move equality: compare by policy index (game-agnostic).
    bool move_equals(const Move& a, const Move& b) const {
        // We intentionally avoid calling Traits::move_to_policy_idx here
        // because we need no position context.  Instead rely on Move's own
        // equality operator which all Traits types define.
        return a == b;
    }

    void stop_all_trees(int n) {
        // Use cancel() instead of stop() to handle the race where the worker
        // thread hasn't started search() yet when we try to stop it.
        // cancel() prevents search() from running and immediately returns an
        // empty result if called before search() starts.
        for (int i = 0; i < n; ++i) {
            auto& t = trees_[static_cast<std::size_t>(i)];
            if (t.mcts) t.mcts->cancel();
        }
        for (int i = 0; i < n; ++i) {
            auto& t = trees_[static_cast<std::size_t>(i)];
            if (t.worker.joinable()) t.worker.join();
        }
    }

    // Centipawn conversion (same formula as uci_loop.hpp).
    static int value_to_cp_internal(float v) noexcept {
        if (v >  0.999f) v =  0.999f;
        if (v < -0.999f) v = -0.999f;
        const double cp = 400.0 * std::tan(
            static_cast<double>(v) * 3.14159265358979323846 * 0.5);
        const int clipped = static_cast<int>(std::round(
            std::max(-32000.0, std::min(32000.0, cp))));
        return clipped;
    }

    // Convert Move to its string representation (e.g. UCI move).
    // Delegates to Traits; defined in chess_traits / shogi_traits.
    static std::string move_to_str(const Move& m) {
        // Use whatever the Traits provides as a to-string helper.
        // For chess: ChessMove::to_uci(); for shogi: ShogiMove::to_usi().
        return Traits::move_to_string(m);
    }

    // -----------------------------------------------------------------------
    // Progress emitter thread
    //
    // Uses a condition variable so that stop() wakes it up immediately rather
    // than waiting for the full 500ms sleep interval.
    // -----------------------------------------------------------------------
    void progress_emitter_loop(int n) {
        while (true) {
            // Wait up to 500ms or until ponder_stopped_ is set.
            {
                std::unique_lock<std::mutex> lk(progress_mtx_);
                progress_cv_.wait_for(lk, std::chrono::milliseconds(500),
                    [this] {
                        return ponder_stopped_.load(std::memory_order_acquire);
                    });
            }
            if (ponder_stopped_.load(std::memory_order_acquire)) break;

            json trees_arr = json::array();
            for (int i = 0; i < n; ++i) {
                const auto& t = trees_[static_cast<std::size_t>(i)];
                if (!t.running.load(std::memory_order_relaxed)) continue;

                json te;
                te["tree"]          = i;
                te["opponent_move"] = Traits::move_to_string(t.opp_move);
                te["depth"]         = t.seldepth.load(std::memory_order_relaxed);
                te["score_cp"]      = value_to_cp_internal(
                    t.value.load(std::memory_order_relaxed));
                te["nodes"]         = static_cast<uint64_t>(
                    t.nodes.load(std::memory_order_relaxed));

                {
                    std::lock_guard<std::mutex> lk(t.pv_mtx);
                    te["pv"] = json(t.top_pv);
                }
                trees_arr.push_back(te);
            }

            json notif;
            notif["jsonrpc"] = "2.0";
            notif["method"]  = "ponder_progress";
            notif["params"]["trees"] = trees_arr;
            if (on_event_) on_event_(notif);
        }
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    std::shared_ptr<nn::NNBackend>   backend_;
    int                              threads_per_tree_;
    std::size_t                      input_size_;
    std::size_t                      policy_size_;

    // Single owning Batcher shared across all trees.
    nn::Batcher                      shared_batcher_;

    // Fixed-size array (avoids requiring move-constructibility of PonderTree).
    std::array<PonderTree<Traits>, kMaxTrees> trees_;
    std::array<Position, kMaxTrees>           opp_positions_;
    int                                       active_trees_{0};

    std::function<void(json)>        on_event_;
    std::atomic<bool>                ponder_stopped_{true};
    std::thread                      progress_thread_;
    std::mutex                       progress_mtx_;
    std::condition_variable          progress_cv_;

    std::atomic<int>                 promoted_tree_idx_{-1};
    std::unique_ptr<MCTSType>        promoted_mcts_;
    Position                         promoted_position_;
};

} // namespace chess_mlx::search
