// SPDX-License-Identifier: MIT
// engine/include/search/mcts.hpp
//
// Template class MCTS<Traits> — PUCT-based Monte Carlo tree search that plugs
// into either ChessTraits or ShogiTraits via the GameRules<Traits> contract.
//
// Features:
//   * Flat, contiguous node pool (std::vector<Node>) for cache friendliness.
//   * Multi-threaded with virtual loss and atomic node visits.
//   * NN evaluations funnelled through nn::Batcher → nn::NNBackend.
//   * FPU reduction, PUCT exploration, Dirichlet noise on the root.
//   * Time/nodes/depth time-control via a TimeControl struct.
//   * Supports MultiPV (k > 1) output for GUI display.
//
// Design notes:
//   * Search state is owned by the MCTS instance; `search()` is the public
//     entry point, thread-safe against concurrent `stop()` calls.
//   * A per-search tree is held in `nodes_`.  Subsequent `search()` calls on
//     the same position reuse the existing tree when the root position hash
//     matches; otherwise the tree is rebuilt.

#pragma once

#include "core/game_rules.hpp"
#include "nn/backend.hpp"
#include "nn/batcher.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace chess_mlx::search {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
inline constexpr float kCPuct          = 1.25f;
inline constexpr float kVirtualLoss    = 1.0f;
inline constexpr float kFpuReduction   = 0.5f;
inline constexpr float kDirichletAlphaChess = 0.30f;
inline constexpr float kDirichletAlphaShogi = 0.15f;
inline constexpr float kDirichletEps   = 0.25f;

// ---------------------------------------------------------------------------
// Time control
// ---------------------------------------------------------------------------
struct TimeControl {
    // If >= 0, hard wall-clock limit in milliseconds.
    int movetime_ms { -1 };
    // If >= 0, search to at most this many nodes.
    int max_nodes   { -1 };
    // If >= 0, search to at most this depth (selection depth).
    int max_depth   { -1 };
    // If true, run until stop().
    bool infinite   { false };
    // Optional: wtime/btime/winc/binc in ms — used by helper
    // `derive_movetime()` to estimate a sensible movetime.
    int wtime_ms { -1 }, btime_ms { -1 }, winc_ms { -1 }, binc_ms { -1 };
    int movestogo { -1 };
    // If true, the position is side-to-move white (for wtime vs btime choice).
    bool white_to_move { true };

    // Compute a sensible movetime if only clocks were supplied.  Very simple
    // time manager: `my_time / (moves_to_go or 30) + my_inc * 0.8`, clamped.
    [[nodiscard]] int derive_movetime_ms() const {
        if (movetime_ms >= 0)       return movetime_ms;
        if (infinite || max_nodes >= 0 || max_depth >= 0) return -1;

        const int my_time = white_to_move ? wtime_ms : btime_ms;
        const int my_inc  = white_to_move ? winc_ms  : binc_ms;
        if (my_time <= 0) return -1;
        const int divisor = (movestogo > 0) ? (movestogo + 1) : 30;
        int base = my_time / divisor;
        if (my_inc > 0) base += (my_inc * 80) / 100;
        // Keep ≥ 10 ms, ≤ my_time / 2.
        if (base < 10) base = 10;
        if (base > my_time / 2) base = my_time / 2;
        return base;
    }
};

// ---------------------------------------------------------------------------
// Node — kept small for cache friendliness.  SoA representation using parallel
// arrays would be marginally tighter; we use AoS for code clarity.  Size: 24
// bytes on 64-bit (uint32 + uint32 + atomic<uint32> + float + float + int32).
// The atomic counters push the size a bit.
// ---------------------------------------------------------------------------
struct Node {
    // Index into `nodes_` of the first child.  0 = no children yet.
    std::uint32_t first_child_idx{0};
    // Number of children.
    std::uint16_t num_children{0};
    // Flag bits.
    std::uint16_t flags{0};          // bit0 = expanded, bit1 = terminal
    // Visit count (atomic for multi-threaded updates).
    std::atomic<std::uint32_t> visits{0};
    // Virtual-loss count (atomic).
    std::atomic<std::uint32_t> virtual_loss{0};
    // Sum of backpropped values from this node's perspective (atomic via mutex).
    std::atomic<int64_t> value_sum_fp{0};  // stored as fixed-point int64, *1e6
    // Prior probability from the parent's NN policy.
    float prior{0.0f};
    // Move index that leads into this node (from parent).
    std::int32_t move_policy_idx{-1};

    Node() = default;
    // Atomics are not copyable — disable copy + move.
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&& o) noexcept
        : first_child_idx(o.first_child_idx),
          num_children(o.num_children),
          flags(o.flags),
          visits(o.visits.load()),
          virtual_loss(o.virtual_loss.load()),
          value_sum_fp(o.value_sum_fp.load()),
          prior(o.prior),
          move_policy_idx(o.move_policy_idx) {}

    static constexpr int64_t kFpScale = 1000000;
    void   add_value(float v)      { value_sum_fp.fetch_add(static_cast<int64_t>(v * kFpScale), std::memory_order_relaxed); }
    double avg_value() const {
        const std::uint32_t n = visits.load(std::memory_order_relaxed);
        if (n == 0) return 0.0;
        return static_cast<double>(value_sum_fp.load(std::memory_order_relaxed))
             / (kFpScale * static_cast<double>(n));
    }

    bool is_expanded() const { return (flags & 1u) != 0; }
    void set_expanded()      { flags = static_cast<std::uint16_t>(flags | 1u); }
    bool is_terminal() const { return (flags & 2u) != 0; }
    void set_terminal()      { flags = static_cast<std::uint16_t>(flags | 2u); }
};

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------
template <typename Traits>
struct SearchResult {
    using Move = typename Traits::Move;

    Move                                  best_move{};
    float                                 root_value{0.0f};      // [-1, +1], stm-relative
    int                                   seldepth{0};
    int                                   avg_depth{0};
    std::uint64_t                         nodes{0};
    std::uint64_t                         elapsed_ms{0};
    // Top principal variations for MultiPV output.
    std::vector<std::vector<Move>>        top_pvs;
    std::vector<float>                    top_values;    // matched to top_pvs
    std::vector<std::uint32_t>            top_visits;
};

// ---------------------------------------------------------------------------
// MCTS<Traits>
// ---------------------------------------------------------------------------
template <typename Traits>
class MCTS {
public:
    using Rules    = core::GameRules<Traits>;
    using Position = typename Traits::Position;
    using Move     = typename Traits::Move;

    struct Config {
        int                                 threads{1};
        int                                 multipv{1};
        bool                                add_root_noise{false};
        float                               dirichlet_alpha{kDirichletAlphaChess};
        // Size in bytes of the input encoding tensor.
        std::size_t                         input_size{64 * 19};
        // Size of the policy vector returned by the NN.
        std::size_t                         policy_size{4672};
        // Called every ~250 ms during search, for `info` lines.
        // Pass nullptr to disable.
        std::function<void(const SearchResult<Traits>&)>
                                            progress_callback{};
        std::chrono::milliseconds           progress_interval{250};
    };

    // Max nodes the tree can hold.  Allocated once; prevents pointer
    // invalidation when children are added concurrently.  ~48 MB for 1M
    // nodes @ ~48 bytes each with padding.
    static constexpr std::size_t kMaxNodes = 1u << 20;

    // Standard owning constructor: creates its own Batcher.
    MCTS(std::shared_ptr<nn::NNBackend> backend, Config cfg)
        : cfg_(std::move(cfg)),
          backend_(std::move(backend)),
          batcher_(backend_) {
        nodes_.reserve(kMaxNodes);
    }

    // Shared-batcher constructor: uses the provided external Batcher for all
    // leaf evaluations.  The caller (MultiPonderManager) owns the Batcher and
    // must ensure it outlives this MCTS instance.  `shared_batcher` must not
    // be null.
    MCTS(std::shared_ptr<nn::NNBackend> backend, Config cfg,
         nn::Batcher* shared_batcher)
        : cfg_(std::move(cfg)),
          backend_(std::move(backend)),
          batcher_(shared_batcher) {
        nodes_.reserve(kMaxNodes);
    }

    ~MCTS() {
        stop();
        // Only shut down the batcher if it is owning (non-shared).
        if (!batcher_.is_shared()) batcher_.shutdown();
    }

    // Reset the tree (e.g. on ucinewgame).
    void reset() {
        std::lock_guard<std::mutex> lk(tree_mtx_);
        nodes_.clear();
        nodes_.reserve(kMaxNodes);
        have_root_ = false;
        total_nodes_.store(0, std::memory_order_relaxed);
    }

    // Request the search to halt ASAP.  Safe to call from any thread.
    void stop() {
        {
            std::lock_guard<std::mutex> lk(stop_mtx_);
            stop_.store(true, std::memory_order_release);
        }
        stop_cv_.notify_all();  // wake any sleeping reporter thread
    }

    // Cancel all future searches.  Once cancelled, search() returns an empty
    // result immediately.  Call reset_cancel() before starting a new search.
    void cancel() {
        {
            std::lock_guard<std::mutex> lk(stop_mtx_);
            stop_.store(true, std::memory_order_release);
            cancelled_.store(true, std::memory_order_release);
        }
        stop_cv_.notify_all();
    }

    // Lift the cancel (call before a new search).
    void reset_cancel() {
        std::lock_guard<std::mutex> lk(stop_mtx_);
        cancelled_.store(false, std::memory_order_release);
    }

    // Configure a callback invoked periodically during search (e.g. to emit
    // UCI `info` lines).
    void set_progress_callback(std::function<void(const SearchResult<Traits>&)> cb) {
        cfg_.progress_callback = std::move(cb);
    }

    // Set MultiPV count.
    void set_multipv(int k) { cfg_.multipv = std::max(1, k); }

    // Set thread count.
    void set_threads(int t) { cfg_.threads = std::max(1, t); }

    // Run a search and return the final result.  Blocks until the search
    // terminates by time, nodes, depth, or stop().
    SearchResult<Traits> search(const Position& pos, const TimeControl& tc) {
        // If this MCTS was cancelled before search() was called (e.g. because
        // stop_all_trees() ran before the worker thread reached search()),
        // return immediately with an empty result.
        {
            std::lock_guard<std::mutex> lk(stop_mtx_);
            if (cancelled_.load(std::memory_order_acquire)) {
                return SearchResult<Traits>{};
            }
            stop_.store(false, std::memory_order_release);
        }
        start_time_ = std::chrono::steady_clock::now();
        tc_ = tc;
        deadline_ms_ = tc.derive_movetime_ms();

        // Reset counters for a new search.
        total_nodes_.store(0, std::memory_order_relaxed);

        // Re-use tree across consecutive calls on the same position hash.
        const std::uint64_t h = Traits::hash(pos);
        {
            std::lock_guard<std::mutex> lk(tree_mtx_);
            if (!have_root_ || root_hash_ != h) {
                nodes_.clear();
                nodes_.reserve(kMaxNodes);
                nodes_.emplace_back();
                root_hash_ = h;
                have_root_ = true;
                root_expanded_ = false;
            }
        }

        // Expand root (synchronous — needed before threads can descend).
        {
            Position root_copy = pos;
            expand_node(0, root_copy);
            if (cfg_.add_root_noise) apply_dirichlet_noise(0);
            root_expanded_ = true;
        }

        // Launch workers.
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(cfg_.threads));
        for (int i = 0; i < cfg_.threads; ++i) {
            workers.emplace_back([this, &pos]() {
                worker_loop(pos);
            });
        }

        // Reporter thread — emits progress callbacks.
        std::thread reporter;
        if (cfg_.progress_callback) {
            reporter = std::thread([this, &pos]() {
                reporter_loop(pos);
            });
        }

        for (auto& t : workers) t.join();
        stop_.store(true, std::memory_order_release);
        if (reporter.joinable()) reporter.join();

        return build_result(pos);
    }

    // Inspect the current node count (for testing / reporting).
    std::size_t node_count() const { return nodes_.size(); }

    // Snapshot the current best result without terminating the search.
    // Returns an empty result if no search has been run yet.
    SearchResult<Traits> snapshot(const Position& root_pos) {
        if (!have_root_) return {};
        return build_result(root_pos);
    }

    // Total nodes accumulated since last reset / search start.
    std::uint64_t total_nodes() const {
        return total_nodes_.load(std::memory_order_relaxed);
    }

    // Batcher stats for diagnostics.
    std::uint64_t batcher_total_batches()   const { return batcher_.total_batches();  }
    std::uint64_t batcher_total_requests()  const { return batcher_.total_requests(); }
    double        batcher_avg_batch_size()  const {
        const auto b = batcher_.total_batches();
        const auto r = batcher_.total_requests();
        return (b > 0) ? static_cast<double>(r) / static_cast<double>(b) : 0.0;
    }

    // Promote a child move as the new root.
    //
    // Performs a BFS-based subtree compaction so that the subtree rooted at
    // `move`'s child becomes node[0].  All relative child indices are
    // rewritten.  This is O(subtree_size) — acceptable at ponder-hit time.
    //
    // Returns true on success.  Returns false if `move` is not found among
    // root's children (caller must fall back to a fresh search).
    bool promote_root(const Position& current_pos, const Move& move) {
        std::lock_guard<std::mutex> lk(tree_mtx_);
        if (!have_root_ || nodes_.empty()) return false;
        if (!nodes_[0].is_expanded() || nodes_[0].num_children == 0) return false;

        const int target_policy = Traits::move_to_policy_idx(move, current_pos);

        // Find matching child index (absolute).
        std::uint32_t target_abs = std::numeric_limits<std::uint32_t>::max();
        {
            const Node& root = nodes_[0];
            for (std::uint16_t i = 0; i < root.num_children; ++i) {
                const std::uint32_t abs = root.first_child_idx + i;
                if (nodes_[abs].move_policy_idx == target_policy) {
                    target_abs = abs;
                    break;
                }
            }
        }
        if (target_abs == std::numeric_limits<std::uint32_t>::max()) return false;

        // BFS compaction: remap old absolute indices to new sequential ones.
        // new_nodes[0] will be the promoted child.
        std::vector<Node> new_nodes;
        new_nodes.reserve(nodes_.capacity());

        // old_to_new[old_idx] = new_idx.  We use a flat array; indices not
        // in the subtree are never accessed.
        std::vector<std::uint32_t> old_to_new(nodes_.size(),
            std::numeric_limits<std::uint32_t>::max());

        // BFS queue of old absolute indices to process.
        std::vector<std::uint32_t> bfs_queue;
        bfs_queue.reserve(nodes_.size());
        bfs_queue.push_back(target_abs);

        while (!bfs_queue.empty()) {
            // Pop front (BFS order preserves parent-before-child invariant).
            const std::uint32_t old_idx = bfs_queue.front();
            bfs_queue.erase(bfs_queue.begin());

            const std::uint32_t new_idx =
                static_cast<std::uint32_t>(new_nodes.size());
            old_to_new[old_idx] = new_idx;

            // Move-construct into new_nodes (atomics require explicit load/store).
            new_nodes.emplace_back();
            Node& dst = new_nodes.back();
            const Node& src = nodes_[old_idx];
            dst.visits.store(src.visits.load());
            dst.virtual_loss.store(0);  // reset virtual loss on promotion
            dst.value_sum_fp.store(src.value_sum_fp.load());
            dst.prior            = src.prior;
            dst.move_policy_idx  = src.move_policy_idx;
            dst.flags            = src.flags;
            dst.num_children     = src.num_children;
            dst.first_child_idx  = src.first_child_idx;  // will be patched below

            // Enqueue children.
            if (src.is_expanded() && src.num_children > 0) {
                for (std::uint16_t c = 0; c < src.num_children; ++c) {
                    bfs_queue.push_back(src.first_child_idx + c);
                }
            }
        }

        // Patch first_child_idx references in new_nodes using old_to_new map.
        for (auto& n : new_nodes) {
            if (!n.is_expanded() || n.num_children == 0) continue;
            const std::uint32_t old_first = n.first_child_idx;
            if (old_first < old_to_new.size()
                && old_to_new[old_first] != std::numeric_limits<std::uint32_t>::max()) {
                n.first_child_idx = old_to_new[old_first];
            } else {
                // Children not captured in BFS (should not happen).
                n.num_children   = 0;
                n.first_child_idx = 0;
                n.flags = static_cast<std::uint16_t>(n.flags & ~1u);  // unexpand
            }
        }

        // Root has no parent move.
        if (!new_nodes.empty()) new_nodes[0].move_policy_idx = -1;

        nodes_     = std::move(new_nodes);
        have_root_ = true;
        root_hash_ = 0;  // invalidated; will be re-set on next search() call
        return true;
    }

private:
    // -----------------------------------------------------------------------
    // PendingLeaf — captures the state of one in-flight NN evaluation.
    // -----------------------------------------------------------------------
    struct PendingLeaf {
        // Node path from root to leaf (inclusive).
        std::vector<std::uint32_t> path;
        // Game position at the leaf (for move-gen during expansion).
        Position                   pos;
        // Leaf node index.
        std::uint32_t              leaf_idx{0};
        // Future from the batcher (invalid if terminal — value is set directly).
        std::future<nn::NNOutput>  fut;
        // For terminal leaves the value is already known; skip fut.get().
        bool                       is_terminal_leaf{false};
        float                      terminal_value{0.0f};
        // Already expanded by another thread before we could submit.
        bool                       already_expanded{false};
        float                      already_value{0.0f};
        // Encoded input tensor submitted to the batcher (kept so we can pass
        // legal-move info to expand_node_with_output).
        std::vector<float>         input_enc;
        // Legal moves generated at the leaf (needed for expansion after fut resolves).
        // We pre-generate them at select time to avoid regenerating under the lock.
        std::vector<typename Traits::Move> legal_moves;
        std::vector<int>                   policy_idxs;
    };

    // -----------------------------------------------------------------------
    // Selection phase: walk from root to a leaf, apply virtual loss on the
    // path, encode + submit the NN evaluation if needed.
    // Returns a fully-populated PendingLeaf.
    // -----------------------------------------------------------------------
    PendingLeaf select_leaf(const Position& root_pos) {
        PendingLeaf pl;
        pl.pos = root_pos;
        pl.path.reserve(64);
        pl.path.push_back(0);

        std::uint32_t cur = 0;
        while (true) {
            const Node& n = nodes_[cur];
            if (n.is_terminal()) break;
            if (!n.is_expanded() || n.num_children == 0) break;

            const std::uint32_t child_local = select_child_puct(cur);
            if (child_local == std::numeric_limits<std::uint32_t>::max()) break;
            const std::uint32_t child_idx =
                nodes_[cur].first_child_idx + child_local;

            const int p_idx = nodes_[child_idx].move_policy_idx;
            Move m;
            if (p_idx >= 0) m = Traits::policy_idx_to_move(p_idx, pl.pos);
            Rules::apply(pl.pos, m);

            nodes_[child_idx].virtual_loss.fetch_add(1, std::memory_order_relaxed);
            pl.path.push_back(child_idx);
            cur = child_idx;
        }
        pl.leaf_idx = cur;

        // Check terminal.
        if (nodes_[cur].is_terminal()) {
            pl.is_terminal_leaf = true;
            pl.terminal_value   = terminal_stm_value(pl.pos);
            return pl;
        }
        if (Rules::is_terminal(pl.pos)) {
            nodes_[cur].set_terminal();
            pl.is_terminal_leaf = true;
            pl.terminal_value   = terminal_stm_value(pl.pos);
            return pl;
        }

        // Early-out: another thread already expanded this node.
        if (nodes_[cur].is_expanded()) {
            pl.already_expanded = true;
            pl.already_value    = static_cast<float>(nodes_[cur].avg_value());
            return pl;
        }

        // Encode input tensor and generate legal moves.
        pl.input_enc.assign(cfg_.input_size, 0.0f);
        Traits::encode_nn(pl.pos, pl.input_enc.data());

        core::MoveList<Move> ml;
        Traits::generate_legal(pl.pos, ml);

        if (ml.empty()) {
            // No legal moves → terminal (stalemate / checkmate).
            {
                std::lock_guard<std::mutex> lk(tree_mtx_);
                if (!nodes_[cur].is_expanded()) {
                    nodes_[cur].set_expanded();
                    nodes_[cur].set_terminal();
                }
            }
            pl.is_terminal_leaf = true;
            pl.terminal_value   = terminal_stm_value(pl.pos);
            return pl;
        }

        for (std::size_t i = 0; i < ml.size(); ++i) {
            pl.legal_moves.push_back(ml[i]);
            pl.policy_idxs.push_back(Traits::move_to_policy_idx(ml[i], pl.pos));
        }

        // Submit to batcher (or direct evaluate if single-threaded / no batch).
        if (cfg_.threads > 1 && backend_->batch_preferred()) {
            pl.fut = batcher_.submit(pl.input_enc);
        }
        // else: we'll call backend_->evaluate() synchronously in drain_leaf().

        return pl;
    }

    // -----------------------------------------------------------------------
    // Drain phase: wait for the future (if any), install children, backprop.
    // -----------------------------------------------------------------------
    void drain_leaf(PendingLeaf& pl, const Position& /*root_pos*/) {
        float leaf_value = 0.0f;

        if (pl.is_terminal_leaf) {
            leaf_value = pl.terminal_value;
        } else if (pl.already_expanded) {
            leaf_value = pl.already_value;
        } else {
            // Get NN output.
            nn::NNOutput nn_out;
            if (cfg_.threads > 1 && backend_->batch_preferred()) {
                nn_out = pl.fut.get();
            } else {
                nn_out = backend_->evaluate(pl.input_enc);
            }

            // Build prior array from NN policy.
            const std::size_t n_moves = pl.legal_moves.size();
            std::vector<float> priors(n_moves, 0.0f);
            float max_logit = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < n_moves; ++i) {
                const int p = pl.policy_idxs[i];
                const float l = (p >= 0 &&
                    static_cast<std::size_t>(p) < nn_out.policy.size())
                    ? nn_out.policy[static_cast<std::size_t>(p)] : 0.0f;
                priors[i] = l;
                if (l > max_logit) max_logit = l;
            }
            double sum = 0.0;
            for (auto& p : priors) { p = std::exp(p - max_logit); sum += p; }
            if (sum > 0.0) {
                for (auto& p : priors) p = static_cast<float>(static_cast<double>(p) / sum);
            } else {
                const float uni = 1.0f / static_cast<float>(n_moves);
                for (auto& p : priors) p = uni;
            }

            // Install children under tree lock.
            {
                std::lock_guard<std::mutex> lk(tree_mtx_);
                if (!nodes_[pl.leaf_idx].is_expanded()) {
                    if (nodes_.size() + n_moves <= nodes_.capacity()) {
                        const std::uint32_t first =
                            static_cast<std::uint32_t>(nodes_.size());
                        nodes_.resize(nodes_.size() + n_moves);
                        for (std::size_t i = 0; i < n_moves; ++i) {
                            Node& c = nodes_[first + i];
                            c.prior           = priors[i];
                            c.move_policy_idx = pl.policy_idxs[i];
                        }
                        nodes_[pl.leaf_idx].first_child_idx = first;
                        nodes_[pl.leaf_idx].num_children =
                            static_cast<std::uint16_t>(n_moves);
                    }
                    nodes_[pl.leaf_idx].set_expanded();
                }
            }
            leaf_value = nn_out.value;
        }

        // Backprop.
        float v = leaf_value;
        for (auto it = pl.path.rbegin(); it != pl.path.rend(); ++it) {
            const auto idx = *it;
            nodes_[idx].visits.fetch_add(1, std::memory_order_relaxed);
            nodes_[idx].add_value(v);
            if (idx != 0) {
                nodes_[idx].virtual_loss.fetch_sub(1, std::memory_order_relaxed);
            }
            v = -v;
        }

        // Node-count termination.
        const std::uint64_t n =
            total_nodes_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (tc_.max_nodes >= 0 &&
            n >= static_cast<std::uint64_t>(tc_.max_nodes)) {
            stop_.store(true, std::memory_order_release);
        }
    }

    // -----------------------------------------------------------------------
    // Worker loop — pipelined: keeps kPipelineDepth leaves in-flight
    // simultaneously so the GPU sees batches of (threads × kPipelineDepth).
    // -----------------------------------------------------------------------

    // Compute per-thread pipeline depth based on thread count.
    // Target: threads × pipeline_depth ≈ 16 total simultaneous virtual losses.
    // This keeps the effective GPU batch size high while preventing all threads
    // from saturating every promising path with virtual loss.
    static int compute_pipeline_depth(int threads) {
        if (threads <= 0) return 1;
        const int target_total_vl = 16;
        int depth = target_total_vl / threads;
        if (depth < 1) depth = 1;
        if (depth > 8) depth = 8;
        return depth;
    }

    void worker_loop(const Position& root_pos) {
        // If single-threaded or backend prefers synchronous, fall back to simple loop.
        if (cfg_.threads <= 1 || !backend_->batch_preferred()) {
            worker_loop_simple(root_pos);
            return;
        }

        // Adaptive pipeline depth: keeps total simultaneous VL ≈ 16 regardless
        // of thread count.  E.g., 4 threads → depth=4 (16 total);
        // 8 threads → depth=2 (16 total); 16 threads → depth=1 (16 total).
        const int pipeline_depth = compute_pipeline_depth(cfg_.threads);

        // Pipelined multi-leaf loop.
        // Use a deque for O(1) front removal.
        std::deque<PendingLeaf> pending;

        // Helper: drain any synchronous (terminal / already-expanded) leaves
        // from the back of the queue immediately, avoiding pipeline stall.
        auto drain_sync_back = [&]() {
            while (!pending.empty() &&
                   (pending.back().is_terminal_leaf ||
                    pending.back().already_expanded)) {
                drain_leaf(pending.back(), root_pos);
                pending.pop_back();
            }
        };

        // Prime the pipeline: submit pipeline_depth async leaves.
        while (!should_stop() && static_cast<int>(pending.size()) < pipeline_depth) {
            pending.push_back(select_leaf(root_pos));
            drain_sync_back();
        }

        // Steady-state: drain oldest, submit new.
        while (!should_stop() || !pending.empty()) {
            if (pending.empty()) break;

            // Drain the front (oldest) pending leaf — blocks until future ready.
            drain_leaf(pending.front(), root_pos);
            pending.pop_front();

            // Submit a new leaf to keep the pipeline full (if not stopping).
            if (!should_stop()) {
                pending.push_back(select_leaf(root_pos));
                drain_sync_back();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Simple (non-pipelined) worker loop — used when threads=1 or backend
    // does not prefer batching.
    // -----------------------------------------------------------------------
    void worker_loop_simple(const Position& root_pos) {
        while (!should_stop()) {
            Position pos = root_pos;
            std::vector<std::uint32_t> path; path.reserve(64);
            path.push_back(0);

            // 1. Selection
            std::uint32_t cur = 0;
            while (true) {
                const Node& n = nodes_[cur];
                if (n.is_terminal()) break;
                if (!n.is_expanded() || n.num_children == 0) break;

                const std::uint32_t child_local = select_child_puct(cur);
                if (child_local == std::numeric_limits<std::uint32_t>::max()) break;
                const std::uint32_t child_idx =
                    nodes_[cur].first_child_idx + child_local;

                const int p_idx = nodes_[child_idx].move_policy_idx;
                Move m;
                if (p_idx >= 0) m = Traits::policy_idx_to_move(p_idx, pos);
                Rules::apply(pos, m);

                nodes_[child_idx].virtual_loss.fetch_add(1, std::memory_order_relaxed);
                path.push_back(child_idx);
                cur = child_idx;
            }

            // 2. Expand & evaluate
            float leaf_value = 0.0f;
            if (nodes_[cur].is_terminal()) {
                leaf_value = terminal_stm_value(pos);
            } else if (Rules::is_terminal(pos)) {
                nodes_[cur].set_terminal();
                leaf_value = terminal_stm_value(pos);
            } else {
                leaf_value = expand_node(cur, pos);
            }

            // 3. Backprop
            float v = leaf_value;
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                auto idx = *it;
                nodes_[idx].visits.fetch_add(1, std::memory_order_relaxed);
                nodes_[idx].add_value(v);
                if (idx != 0) {
                    nodes_[idx].virtual_loss.fetch_sub(1, std::memory_order_relaxed);
                }
                v = -v;
            }

            const std::uint64_t n =
                total_nodes_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (tc_.max_nodes >= 0 &&
                n >= static_cast<std::uint64_t>(tc_.max_nodes)) {
                stop_.store(true, std::memory_order_release);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Reporter loop
    // -----------------------------------------------------------------------
    void reporter_loop(const Position& pos) {
        while (!should_stop()) {
            // Use condition variable so stop() wakes us immediately.
            {
                std::unique_lock<std::mutex> lk(stop_mtx_);
                stop_cv_.wait_for(lk, cfg_.progress_interval,
                    [this] { return stop_.load(std::memory_order_acquire); });
            }
            if (should_stop()) break;
            auto res = build_result(pos);
            if (cfg_.progress_callback) cfg_.progress_callback(res);
        }
    }

    // -----------------------------------------------------------------------
    // Stop checks
    // -----------------------------------------------------------------------
    bool should_stop() const {
        if (stop_.load(std::memory_order_acquire)) return true;
        if (cancelled_.load(std::memory_order_acquire)) return true;
        if (deadline_ms_ >= 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time_).count();
            if (elapsed >= deadline_ms_) return true;
        }
        if (tc_.max_nodes >= 0 &&
            total_nodes_.load(std::memory_order_relaxed) >=
                static_cast<std::uint64_t>(tc_.max_nodes)) {
            return true;
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // PUCT child selection
    // -----------------------------------------------------------------------
    std::uint32_t select_child_puct(std::uint32_t parent_idx) {
        const Node& parent = nodes_[parent_idx];
        const std::uint16_t nc = parent.num_children;
        if (nc == 0) return std::numeric_limits<std::uint32_t>::max();

        const std::uint32_t N_parent = parent.visits.load(std::memory_order_relaxed);
        const double sqrt_n = std::sqrt(static_cast<double>(std::max<std::uint32_t>(1, N_parent)));

        // FPU reduction: compute sum of visited priors.
        double sum_p_visited = 0.0;
        for (std::uint16_t i = 0; i < nc; ++i) {
            const Node& c = nodes_[parent.first_child_idx + i];
            if (c.visits.load(std::memory_order_relaxed) > 0)
                sum_p_visited += c.prior;
        }
        const double parent_q = parent.avg_value();
        const double fpu_reduced_q = parent_q - kFpuReduction * std::sqrt(sum_p_visited);

        double best_score = -std::numeric_limits<double>::infinity();
        std::uint32_t best_idx = 0;
        for (std::uint16_t i = 0; i < nc; ++i) {
            const Node& c = nodes_[parent.first_child_idx + i];
            const std::uint32_t Nc = c.visits.load(std::memory_order_relaxed);
            const std::uint32_t vl = c.virtual_loss.load(std::memory_order_relaxed);
            // From parent's perspective, value_sum of child is from child's stm
            // (= parent's opponent) viewpoint, so we negate.
            double q;
            if (Nc + vl == 0) {
                q = fpu_reduced_q;
            } else {
                const double s = static_cast<double>(c.value_sum_fp.load(std::memory_order_relaxed))
                               / Node::kFpScale;
                q = -s / static_cast<double>(Nc + vl);
                // Virtual loss bias: treat the in-flight visits as worst-case.
                // s has been reduced by virtual-loss accumulations on backprop,
                // but we never add vl onto value_sum; instead we just inflate
                // the denominator, which biases Q toward 0 for busy children.
                // That's fine per the standard virtual-loss recipe.
            }
            const double u = kCPuct * static_cast<double>(c.prior)
                             * sqrt_n / (1.0 + static_cast<double>(Nc + vl));
            const double score = q + u;
            if (score > best_score) {
                best_score = score;
                best_idx   = i;
            }
        }
        return best_idx;
    }

    // -----------------------------------------------------------------------
    // Terminal-value helpers
    //
    // Returns the terminal value from the perspective of the side-to-move.
    // For chess, Traits::terminal_value returns white-perspective; flip if
    // black to move.  For shogi, Traits::terminal_value returns stm-perspective
    // directly.
    // -----------------------------------------------------------------------
    float terminal_stm_value(const Position& pos) {
        return Traits::terminal_value(pos);
    }

    // -----------------------------------------------------------------------
    // Node expansion: evaluate NN on leaf, create children.
    // Returns leaf value from stm perspective.
    // -----------------------------------------------------------------------
    float expand_node(std::uint32_t idx, const Position& pos) {
        // Early out if another thread already expanded this node.
        if (nodes_[idx].is_expanded()) return static_cast<float>(nodes_[idx].avg_value());

        // Generate legal moves (cheap; repeated work is OK across racing expands).
        core::MoveList<Move> ml;
        Traits::generate_legal(pos, ml);
        if (ml.empty()) {
            std::lock_guard<std::mutex> lk(tree_mtx_);
            if (!nodes_[idx].is_expanded()) {
                nodes_[idx].set_expanded();
                nodes_[idx].set_terminal();
            }
            return terminal_stm_value(pos);
        }

        // Encode + NN eval.
        std::vector<float> input(cfg_.input_size, 0.0f);
        Traits::encode_nn(pos, input.data());

        nn::NNOutput nn_out;
        if (cfg_.threads > 1 && backend_->batch_preferred()) {
            auto fut = batcher_.submit(std::move(input));
            nn_out = fut.get();
        } else {
            nn_out = backend_->evaluate(input);
        }

        // Extract priors for legal moves; softmax with masking.
        std::vector<float> priors(ml.size(), 0.0f);
        std::vector<int>   idxs  (ml.size(), -1);

        float max_logit = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < ml.size(); ++i) {
            const int p = Traits::move_to_policy_idx(ml[i], pos);
            idxs[i] = p;
            const float l = (p >= 0 && static_cast<std::size_t>(p) < nn_out.policy.size())
                          ? nn_out.policy[static_cast<std::size_t>(p)] : 0.0f;
            priors[i] = l;
            if (l > max_logit) max_logit = l;
        }
        // Numerically stable softmax.
        double sum = 0.0;
        for (auto& p : priors) { p = std::exp(p - max_logit); sum += p; }
        if (sum > 0.0) {
            for (auto& p : priors) p = static_cast<float>(static_cast<double>(p) / sum);
        } else {
            const float uni = 1.0f / static_cast<float>(ml.size());
            for (auto& p : priors) p = uni;
        }

        // Reserve + install children under the tree lock; re-check for races.
        {
            std::lock_guard<std::mutex> lk(tree_mtx_);
            if (nodes_[idx].is_expanded()) return nn_out.value;

            // Capacity guard: if we'd exceed the pool, mark terminal and
            // return NN value.  This caps tree growth rather than crashing.
            if (nodes_.size() + ml.size() > nodes_.capacity()) {
                nodes_[idx].set_expanded();
                return nn_out.value;
            }

            const std::uint32_t first = static_cast<std::uint32_t>(nodes_.size());
            nodes_.resize(nodes_.size() + ml.size());
            for (std::size_t i = 0; i < ml.size(); ++i) {
                Node& c = nodes_[first + i];
                c.prior           = priors[i];
                c.move_policy_idx = idxs[i];
            }
            nodes_[idx].first_child_idx = first;
            nodes_[idx].num_children    = static_cast<std::uint16_t>(ml.size());
            nodes_[idx].set_expanded();
        }

        return nn_out.value;  // NN value is already stm-relative by convention
    }

    // -----------------------------------------------------------------------
    // Dirichlet noise on root (self-play).
    // -----------------------------------------------------------------------
    void apply_dirichlet_noise(std::uint32_t idx) {
        const Node& n = nodes_[idx];
        if (!n.is_expanded() || n.num_children == 0) return;

        std::gamma_distribution<float> gd(cfg_.dirichlet_alpha, 1.0f);
        std::vector<float> g(n.num_children);
        float total = 0.0f;
        for (auto& x : g) { x = gd(rng_); total += x; }
        if (total <= 0.0f) return;
        for (auto& x : g) x /= total;

        for (std::uint16_t i = 0; i < n.num_children; ++i) {
            Node& c = nodes_[n.first_child_idx + i];
            c.prior = (1.0f - kDirichletEps) * c.prior
                    + kDirichletEps * g[i];
        }
    }

    // -----------------------------------------------------------------------
    // Build the final / incremental SearchResult.
    // -----------------------------------------------------------------------
    SearchResult<Traits> build_result(const Position& root_pos) {
        SearchResult<Traits> r;
        r.nodes = total_nodes_.load(std::memory_order_relaxed);
        r.elapsed_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time_).count());

        const Node& root = nodes_[0];
        if (root.num_children == 0) return r;

        // Collect (child index, visits, mean_q, move) tuples.
        struct Candidate {
            std::uint32_t idx;
            std::uint32_t visits;
            double        q;
            Move          move;
        };
        std::vector<Candidate> cands;
        cands.reserve(root.num_children);
        for (std::uint16_t i = 0; i < root.num_children; ++i) {
            const Node& c = nodes_[root.first_child_idx + i];
            Candidate cd;
            cd.idx    = root.first_child_idx + i;
            cd.visits = c.visits.load(std::memory_order_relaxed);
            cd.q      = -c.avg_value();  // child's q from root's perspective
            const int p = c.move_policy_idx;
            cd.move   = (p >= 0) ? Traits::policy_idx_to_move(p, root_pos) : Move{};
            cands.push_back(cd);
        }
        // Sort by visits desc, then by q desc.
        std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
            if (a.visits != b.visits) return a.visits > b.visits;
            return a.q > b.q;
        });

        r.best_move  = cands.front().move;
        r.root_value = static_cast<float>(cands.front().q);

        const int kmax = std::min<int>(cfg_.multipv, static_cast<int>(cands.size()));
        r.top_pvs.reserve(static_cast<std::size_t>(kmax));
        r.top_values.reserve(static_cast<std::size_t>(kmax));
        r.top_visits.reserve(static_cast<std::size_t>(kmax));

        for (int k = 0; k < kmax; ++k) {
            r.top_pvs.emplace_back(build_pv(cands[static_cast<std::size_t>(k)].idx, root_pos,
                                            cands[static_cast<std::size_t>(k)].move));
            r.top_values.push_back(static_cast<float>(cands[static_cast<std::size_t>(k)].q));
            r.top_visits.push_back(cands[static_cast<std::size_t>(k)].visits);
        }

        // Selection-depth estimate: walk best path from root.
        int depth = 0;
        {
            Position tmp = root_pos;
            std::uint32_t cur = 0;
            while (true) {
                const Node& n = nodes_[cur];
                if (!n.is_expanded() || n.num_children == 0) break;
                // Most-visited child.
                std::uint32_t best = 0;
                std::uint32_t bv = 0;
                for (std::uint16_t i = 0; i < n.num_children; ++i) {
                    const Node& c = nodes_[n.first_child_idx + i];
                    const auto v = c.visits.load(std::memory_order_relaxed);
                    if (v > bv) { bv = v; best = i; }
                }
                if (bv == 0) break;
                ++depth;
                const std::uint32_t next = n.first_child_idx + best;
                const int p_idx = nodes_[next].move_policy_idx;
                if (p_idx < 0) break;
                Rules::apply(tmp, Traits::policy_idx_to_move(p_idx, tmp));
                cur = next;
                if (depth >= 256) break;  // safety guard
            }
        }
        r.seldepth = depth;
        r.avg_depth = depth;
        return r;
    }

    // Build a single PV by repeatedly selecting the most-visited child.
    std::vector<Move> build_pv(std::uint32_t start_idx,
                                const Position& root_pos,
                                const Move& first_move) {
        std::vector<Move> pv;
        pv.push_back(first_move);
        Position tmp = root_pos;
        Rules::apply(tmp, first_move);

        std::uint32_t cur = start_idx;
        for (int d = 0; d < 64; ++d) {
            const Node& n = nodes_[cur];
            if (!n.is_expanded() || n.num_children == 0) break;
            std::uint32_t best = 0;
            std::uint32_t bv = 0;
            for (std::uint16_t i = 0; i < n.num_children; ++i) {
                const Node& c = nodes_[n.first_child_idx + i];
                const auto v = c.visits.load(std::memory_order_relaxed);
                if (v > bv) { bv = v; best = i; }
            }
            if (bv == 0) break;
            const std::uint32_t next = n.first_child_idx + best;
            const int p_idx = nodes_[next].move_policy_idx;
            if (p_idx < 0) break;
            Move m = Traits::policy_idx_to_move(p_idx, tmp);
            pv.push_back(m);
            Rules::apply(tmp, m);
            cur = next;
        }
        return pv;
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    Config                         cfg_;
    std::shared_ptr<nn::NNBackend> backend_;
    nn::Batcher                    batcher_;

    std::vector<Node>              nodes_;
    mutable std::mutex             tree_mtx_;
    std::uint64_t                  root_hash_{0};
    bool                           have_root_{false};
    bool                           root_expanded_{false};

    std::atomic<std::uint64_t>     total_nodes_{0};
    std::atomic<bool>              stop_{false};
    std::atomic<bool>              cancelled_{false};
    mutable std::mutex             stop_mtx_;
    std::condition_variable        stop_cv_;

    TimeControl                    tc_;
    int                            deadline_ms_{-1};
    std::chrono::steady_clock::time_point start_time_;

    std::mt19937                   rng_{static_cast<std::uint32_t>(std::random_device{}())};
};

} // namespace chess_mlx::search
