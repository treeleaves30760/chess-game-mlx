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
//
// PUCT exploration constant — replaced the old fixed kCPuct=1.25 with a
// log-growth schedule matching LC0:
//
//   cpuct(N) = kCPuctInit + log((N + kCPuctBase) / kCPuctBase)
//
// At low N the value sits near kCPuctInit (≈1.25, the AlphaZero default).
// As N grows the exploration term scales gently with log(N), which is the
// regime LC0 found to be ~30-60 Elo better than a constant cpuct in real
// games (especially noticeable in deep MCTS where the AlphaZero recipe
// stops exploring quickly enough).  The two-parameter form is LC0's
// `cpuct_init` + `cpuct_base`; we use their published defaults.
inline constexpr float kCPuctInit      = 1.25f;
inline constexpr float kCPuctBase      = 19652.0f;
inline constexpr float kVirtualLoss    = 1.0f;
inline constexpr float kFpuReduction   = 0.5f;
inline constexpr float kDirichletAlphaChess = 0.30f;
inline constexpr float kDirichletAlphaShogi = 0.15f;
inline constexpr float kDirichletEps   = 0.25f;

// Moves-left head bias (LC0 mlh).
//
// When a node has a clearly-winning Q, we prefer children that the NN expects
// to reach the terminal quickly; when losing we prefer children that drag the
// game out.  The standard formula (lc0/src/mcts/search.cc):
//
//   delta_ml = sign(Q) * (child_ml - parent_ml)
//   bonus    = kMlhWeight * clip(Q, -kMlhQThresh, kMlhQThresh)
//                         * clip(delta_ml / kMlhScale, -1, 1)
//
// Only kicks in when |Q| > kMlhActivationThresh, so undecided positions
// behave exactly like before (no bias mixed into the score).  Magnitude is
// kept under kMlhWeight so it never out-shouts the policy/value signal —
// the function of mlh is to break ties between equally-good lines.
inline constexpr float kMlhWeight             = 0.03f;
inline constexpr float kMlhQThresh            = 1.0f;
inline constexpr float kMlhScale              = 20.0f;   // plies
inline constexpr float kMlhActivationThresh   = 0.50f;   // |Q| above this

// Compute the effective cpuct for a parent with N visits.
//
// Implemented as a free helper rather than inlining the formula into
// select_child_puct so that future tuning (e.g. per-game cpuct_base) is a
// single-call-site change, and so the regression test below can exercise
// the schedule directly.
[[nodiscard]] inline double cpuct_at(std::uint32_t N_parent) noexcept {
    // log1p((N + base) / base - 1) == log((N + base) / base) but slightly
    // more accurate for tiny N; doesn't matter at our scales — go with the
    // direct form, which compiles to one log call.
    const double n = static_cast<double>(N_parent);
    return static_cast<double>(kCPuctInit)
         + std::log((n + static_cast<double>(kCPuctBase))
                    /                static_cast<double>(kCPuctBase));
}

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
// arrays would be marginally tighter; we use AoS for code clarity.  The
// atomic counters push the size a bit (~40 bytes on 64-bit).
//
// Memory-ordering protocol (this is correctness-critical with N worker
// threads + virtual loss):
//
//   * `flags` is the publication gate.  When a writer expands a node it
//     does, *under tree_mtx_*:
//         first_child_idx = X;        // plain write
//         num_children    = N;        // plain write
//         set_expanded();             // RELEASE-store: bit 0 = 1
//
//   * Readers in select_leaf / select_child_puct / build_pv etc. do NOT
//     take tree_mtx_ for the fast path:
//         if (is_expanded()) {         // ACQUIRE-load on flags
//             // synchronises-with the writer's release, so the plain
//             // writes to first_child_idx / num_children are visible.
//             read first_child_idx, num_children;
//         }
//
//   * The C++11 release/acquire pair on the same atomic establishes a
//     happens-before edge that covers the non-atomic writes to
//     first_child_idx and num_children.  No extra atomics needed on
//     those fields — they are written exactly once per Node and the
//     publication gate guarantees visibility.
//
//   * is_terminal() / set_terminal() get the same atomic treatment
//     because a reader may observe bit 1 being set during the same
//     race window.
// ---------------------------------------------------------------------------
struct Node {
    // Index into `nodes_` of the first child.  0 = no children yet.
    // Plain because publication is gated by the atomic `flags` (see above).
    std::uint32_t first_child_idx{0};
    // Number of children.  Same publication semantics as first_child_idx.
    std::uint16_t num_children{0};
    // Flag bits.  bit 0 = expanded, bit 1 = terminal.  RELEASE on set,
    // ACQUIRE on read — gates publication of first_child_idx/num_children.
    std::atomic<std::uint16_t> flags{0};
    // Visit count (atomic for multi-threaded updates).
    std::atomic<std::uint32_t> visits{0};
    // Virtual-loss count (atomic).
    std::atomic<std::uint32_t> virtual_loss{0};
    // Sum of backpropped values from this node's perspective (atomic via mutex).
    std::atomic<int64_t> value_sum_fp{0};  // stored as fixed-point int64, *1e6
    // Sum of NN-predicted moves-left, accumulated on backprop.  Stored as
    // fixed-point int64 so updates are lock-free `fetch_add`s.  Average is
    // moves_left_sum_fp / (visits * kMlhFpScale).  Updated alongside
    // value_sum_fp in the backprop loop.
    std::atomic<int64_t> moves_left_sum_fp{0};
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
          flags(o.flags.load(std::memory_order_relaxed)),
          visits(o.visits.load(std::memory_order_relaxed)),
          virtual_loss(o.virtual_loss.load(std::memory_order_relaxed)),
          value_sum_fp(o.value_sum_fp.load(std::memory_order_relaxed)),
          moves_left_sum_fp(o.moves_left_sum_fp.load(std::memory_order_relaxed)),
          prior(o.prior),
          move_policy_idx(o.move_policy_idx) {}

    static constexpr int64_t kFpScale = 1000000;
    // moves_left is a count of plies (typically 0..200), so a 1e3 scale leaves
    // plenty of headroom in int64 even for a million-visit subtree.
    static constexpr int64_t kMlhFpScale = 1000;

    void   add_value(float v)      { value_sum_fp.fetch_add(static_cast<int64_t>(v * kFpScale), std::memory_order_relaxed); }
    void   add_moves_left(float m) {
        moves_left_sum_fp.fetch_add(static_cast<int64_t>(m * kMlhFpScale),
                                    std::memory_order_relaxed);
    }
    double avg_value() const {
        const std::uint32_t n = visits.load(std::memory_order_relaxed);
        if (n == 0) return 0.0;
        return static_cast<double>(value_sum_fp.load(std::memory_order_relaxed))
             / (kFpScale * static_cast<double>(n));
    }
    // Average moves-left from this node's perspective.  Returns -1 when no
    // visits, so callers can detect the "no data yet" state without
    // false-positiving on a genuine `0`.
    double avg_moves_left() const {
        const std::uint32_t n = visits.load(std::memory_order_relaxed);
        if (n == 0) return -1.0;
        return static_cast<double>(moves_left_sum_fp.load(std::memory_order_relaxed))
             / (kMlhFpScale * static_cast<double>(n));
    }

    // Acquire-load: a true result synchronises-with the writer's release in
    // `set_expanded()` so subsequent plain reads of first_child_idx /
    // num_children see the latest values.
    bool is_expanded() const noexcept {
        return (flags.load(std::memory_order_acquire) & 1u) != 0;
    }
    // Release-store: pairs with `is_expanded()`'s acquire so prior plain
    // writes to first_child_idx / num_children become visible.  Caller is
    // expected to be the unique writer for this Node (i.e. holding the
    // tree mutex or guaranteed to be single-threaded at this point).
    void set_expanded() noexcept {
        flags.fetch_or(static_cast<std::uint16_t>(1u),
                       std::memory_order_release);
    }
    bool is_terminal() const noexcept {
        return (flags.load(std::memory_order_acquire) & 2u) != 0;
    }
    void set_terminal() noexcept {
        flags.fetch_or(static_cast<std::uint16_t>(2u),
                       std::memory_order_release);
    }
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
    // Per-candidate proven status, from root's stm perspective:
    //   +1 = the candidate's PV reaches a terminal that wins for root's stm
    //   -1 = the PV reaches a terminal that loses for root's stm
    //    0 = unknown / not yet resolved
    // Used by GUI/UCI layers to emit `score mate N` reliably (without relying
    // on a transient |q| > threshold check that can flicker mid-search).
    std::vector<int>                      top_proven_sign;
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

            // Copy fields explicitly — atomics require load/store, plain
            // members are direct assignment.  We're under tree_mtx_ and the
            // search is stopped, so relaxed memory order is safe everywhere.
            new_nodes.emplace_back();
            Node& dst = new_nodes.back();
            const Node& src = nodes_[old_idx];
            dst.visits      .store(src.visits      .load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
            dst.virtual_loss.store(0u, std::memory_order_relaxed);  // reset on promotion
            dst.value_sum_fp.store(src.value_sum_fp.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
            dst.moves_left_sum_fp.store(src.moves_left_sum_fp.load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
            dst.prior            = src.prior;
            dst.move_policy_idx  = src.move_policy_idx;
            dst.flags           .store(src.flags.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
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
                // Unexpand: clear bit 0.  Single-writer here (we hold
                // tree_mtx_ and no workers run), so relaxed is fine.
                n.flags.fetch_and(static_cast<std::uint16_t>(~1u),
                                  std::memory_order_relaxed);
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
            // Move so the batcher owns the buffer; drain_leaf() doesn't need
            // to read input_enc again once submission is queued.
            pl.fut = batcher_.submit(std::move(pl.input_enc));
        }
        // else: we'll call backend_->evaluate() synchronously in drain_leaf().

        return pl;
    }

    // -----------------------------------------------------------------------
    // Drain phase: wait for the future (if any), install children, backprop.
    // -----------------------------------------------------------------------
    void drain_leaf(PendingLeaf& pl, const Position& /*root_pos*/) {
        float leaf_value      = 0.0f;
        // Terminal leaves resolve in 0 plies; already-expanded leaves reuse
        // the running average we'd compute below if we re-queried.  For both
        // we feed in 0 — the moves-left bias only affects PUCT once a
        // genuine NN estimate has been backpropped through, which is the
        // common case.
        float leaf_moves_left = 0.0f;

        if (pl.is_terminal_leaf) {
            leaf_value = pl.terminal_value;
        } else if (pl.already_expanded) {
            leaf_value = pl.already_value;
            // Recover the running moves_left estimate so the bias keeps
            // pointing in a sensible direction for racing-expansion leaves.
            const double aml = nodes_[pl.leaf_idx].avg_moves_left();
            if (aml >= 0.0) leaf_moves_left = static_cast<float>(aml);
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
                const float l = (p >= 0)
                    ? nn_out.policy_at(static_cast<std::size_t>(p))
                    : 0.0f;
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
            leaf_value      = nn_out.value;
            leaf_moves_left = nn_out.moves_left;
        }

        // Backprop. Order: value_sum then visits. The reporter thread reads
        // avg = value_sum / visits between these two atomic ops; if visits
        // advanced first we'd see value_sum / (n+1) which transiently biases
        // |avg| < 1 even for a proven-win subtree, breaking the mate detection
        // in build_result + uci emit_info. Updating value_sum first means a
        // racing read overshoots (avg = new_sum / old_visits), which is the
        // safer direction for the |avg| >= kProvenWinThreshold check.
        //
        // moves_left is independent — every ancestor sees the same plies-
        // remaining count (it isn't sign-flipped on each ply since "plies
        // until terminal" is the same number from either side's view).
        // We grow it by +1 per ancestor step to reflect the extra move
        // they would each play before reaching the leaf.
        float v  = leaf_value;
        float ml = leaf_moves_left;
        for (auto it = pl.path.rbegin(); it != pl.path.rend(); ++it) {
            const auto idx = *it;
            nodes_[idx].add_value(v);
            nodes_[idx].add_moves_left(ml);
            nodes_[idx].visits.fetch_add(1, std::memory_order_relaxed);
            if (idx != 0) {
                nodes_[idx].virtual_loss.fetch_sub(1, std::memory_order_relaxed);
            }
            v   = -v;
            ml += 1.0f;
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
        //
        // We keep going until should_stop() — we must NOT exit just because the
        // pipeline drained.  In a resolved subtree (e.g. a found forced mate)
        // select_leaf keeps returning terminal leaves that drain_sync_back pops
        // immediately, so `pending` empties even though the search isn't over.
        // Exiting there is the bug that made `go infinite` self-terminate after
        // a few hundred nodes in won endgames (all workers bailed, search ended)
        // and starved the search of the visits it needs to converge on the
        // shortest mate.
        while (!should_stop() || !pending.empty()) {
            if (pending.empty()) {
                if (should_stop()) break;
                // Re-prime.  If every selection resolves to a terminal /
                // already-expanded leaf, drain_sync_back empties them right away
                // and `pending` stays empty; back off briefly so we don't peg
                // every core spinning on a fully-solved position under
                // `go infinite`.
                while (!should_stop() &&
                       static_cast<int>(pending.size()) < pipeline_depth) {
                    pending.push_back(select_leaf(root_pos));
                    drain_sync_back();
                }
                if (pending.empty() && !should_stop()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue;
            }

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
            float leaf_value      = 0.0f;
            float leaf_moves_left = 0.0f;
            if (nodes_[cur].is_terminal()) {
                leaf_value = terminal_stm_value(pos);
            } else if (Rules::is_terminal(pos)) {
                nodes_[cur].set_terminal();
                leaf_value = terminal_stm_value(pos);
            } else {
                leaf_value = expand_node(cur, pos, &leaf_moves_left);
            }

            // 3. Backprop — value_sum first, then visits (see drain_leaf).
            //    moves_left grows by +1 per ancestor (see drain_leaf comment).
            float v  = leaf_value;
            float ml = leaf_moves_left;
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                auto idx = *it;
                nodes_[idx].add_value(v);
                nodes_[idx].add_moves_left(ml);
                nodes_[idx].visits.fetch_add(1, std::memory_order_relaxed);
                if (idx != 0) {
                    nodes_[idx].virtual_loss.fetch_sub(1, std::memory_order_relaxed);
                }
                v   = -v;
                ml += 1.0f;
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
        // Log-growth cpuct: see cpuct_at() / kCPuctInit / kCPuctBase comments
        // at the top of this file.  Computed once per call (depends only on
        // the parent's visit count, not the child's).
        const double cpuct = cpuct_at(N_parent);

        // FPU reduction: compute sum of visited priors.
        double sum_p_visited = 0.0;
        for (std::uint16_t i = 0; i < nc; ++i) {
            const Node& c = nodes_[parent.first_child_idx + i];
            if (c.visits.load(std::memory_order_relaxed) > 0)
                sum_p_visited += c.prior;
        }
        const double parent_q  = parent.avg_value();
        const double parent_ml = parent.avg_moves_left();  // -1 if no visits
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
            const double u = cpuct * static_cast<double>(c.prior)
                             * sqrt_n / (1.0 + static_cast<double>(Nc + vl));

            // -------------------------------------------------------------
            // Moves-left bias (LC0 mlh).  See kMlh* comments at top of file.
            //
            // Active only when Q is decisively non-zero (|Q| > activation)
            // AND the child has at least one visit (so avg_moves_left is
            // meaningful).  Sign convention:
            //
            //   winning  (Q > 0) → prefer SHORT child_ml  → bias negative when child_ml > parent_ml
            //   losing   (Q < 0) → prefer LONG  child_ml  → bias positive when child_ml > parent_ml
            //
            // We frame both cases as a single bias = w * Q * (delta_ml/scale),
            // signed so larger child_ml → smaller score when winning, larger
            // when losing — which falls out of multiplying by Q.
            // -------------------------------------------------------------
            double mlh_bias = 0.0;
            if (Nc > 0 && std::abs(q) > kMlhActivationThresh && parent_ml >= 0.0) {
                const double child_ml = c.avg_moves_left();
                if (child_ml >= 0.0) {
                    double delta = (child_ml - parent_ml) / kMlhScale;
                    if (delta >  1.0) delta =  1.0;
                    if (delta < -1.0) delta = -1.0;
                    double q_clipped = q;
                    if (q_clipped >  kMlhQThresh) q_clipped =  kMlhQThresh;
                    if (q_clipped < -kMlhQThresh) q_clipped = -kMlhQThresh;
                    // -q so that winning + long child_ml → negative bias.
                    mlh_bias = -static_cast<double>(kMlhWeight) * q_clipped * delta;
                }
            }

            const double score = q + u + mlh_bias;
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
    // Returns leaf value from stm perspective.  If `out_moves_left` is
    // non-null, the NN's moves-left estimate is written there (only used by
    // worker_loop_simple — the pipelined drain_leaf path reads moves_left
    // directly from NNOutput).
    // -----------------------------------------------------------------------
    float expand_node(std::uint32_t idx, const Position& pos,
                       float* out_moves_left = nullptr) {
        auto set_ml = [&](float v) { if (out_moves_left) *out_moves_left = v; };

        // Early out if another thread already expanded this node.
        if (nodes_[idx].is_expanded()) {
            const double aml = nodes_[idx].avg_moves_left();
            if (aml >= 0.0) set_ml(static_cast<float>(aml));
            else            set_ml(0.0f);
            return static_cast<float>(nodes_[idx].avg_value());
        }

        // Generate legal moves (cheap; repeated work is OK across racing expands).
        core::MoveList<Move> ml;
        Traits::generate_legal(pos, ml);
        if (ml.empty()) {
            std::lock_guard<std::mutex> lk(tree_mtx_);
            if (!nodes_[idx].is_expanded()) {
                nodes_[idx].set_expanded();
                nodes_[idx].set_terminal();
            }
            set_ml(0.0f);  // terminal: zero plies remaining.
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
            const float l = (p >= 0)
                          ? nn_out.policy_at(static_cast<std::size_t>(p))
                          : 0.0f;
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
            if (nodes_[idx].is_expanded()) {
                set_ml(nn_out.moves_left);
                return nn_out.value;
            }

            // Capacity guard: if we'd exceed the pool, mark terminal and
            // return NN value.  This caps tree growth rather than crashing.
            if (nodes_.size() + ml.size() > nodes_.capacity()) {
                nodes_[idx].set_expanded();
                set_ml(nn_out.moves_left);
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

        set_ml(nn_out.moves_left);
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

        // Hold the tree lock for the whole snapshot.  The reporter thread calls
        // build_result every ~250 ms while workers may be resizing nodes_ (a
        // reallocation that invalidates every reference/index we read here).
        // The per-Node acquire/release protocol guards individual fields but
        // NOT the vector reallocation, so without this lock build_pv's walk can
        // dereference freed memory — a real, if rare, crash, made more likely
        // now that we build a PV for every candidate below.  Workers only hold
        // tree_mtx_ briefly (to install children), so the contention is small.
        std::lock_guard<std::mutex> lk(tree_mtx_);

        const Node& root = nodes_[0];
        if (root.num_children == 0) return r;

        // Collect (child index, visits, mean_q, move, proven info) tuples.
        struct Candidate {
            std::uint32_t      idx;
            std::uint32_t      visits;
            double             q;
            Move               move;
            std::vector<Move>  pv;            // most-visited PV from this child
            int                proven_sign{0};// +1 win / -1 loss / 0 unknown (root stm)
            int                mate_dist{0};  // plies to terminal when proven
        };
        std::vector<Candidate> cands;
        cands.reserve(root.num_children);
        for (std::uint16_t i = 0; i < root.num_children; ++i) {
            const Node& c = nodes_[root.first_child_idx + i];
            const int p = c.move_policy_idx;
            if (p < 0) continue;  // skip unmapped children (defensive: no null PV)
            Candidate cd;
            cd.idx    = root.first_child_idx + i;
            cd.visits = c.visits.load(std::memory_order_relaxed);
            cd.q      = -c.avg_value();  // child's q from root's perspective
            cd.move   = Traits::policy_idx_to_move(p, root_pos);
            // Probe the most-visited path ONCE: yields the proven win/loss flag
            // and, when proven, the distance in plies to the terminal.  Reused
            // for both the sort and the MultiPV output below, so build_pv runs
            // exactly once per candidate.
            cd.pv = build_pv(cd.idx, root_pos, cd.move, &cd.proven_sign);
            cd.mate_dist = (cd.proven_sign != 0)
                ? static_cast<int>(cd.pv.size())
                : std::numeric_limits<int>::max();
            cands.push_back(std::move(cd));
        }
        if (cands.empty()) return r;
        // Sort order, applied top-down:
        //
        //   1. Proven win (q ≥ kProvenWinThreshold) → first.  The subtree is
        //      resolved to a forced win for root's stm and must never be lost
        //      to a non-mating candidate that simply accumulated more visits.
        //   2. Proven loss (q ≤ -kProvenWinThreshold) → last.  These are
        //      forced-loss subtrees the search has resolved; we never want a
        //      proven-loss move to surface as bestmove just because the
        //      policy network sent visits its way before the loss was seen.
        //   3. Q-quality bucket: candidates within `kQBucketDelta` of the top
        //      non-proven Q are treated as "competitive"; non-competitive
        //      moves are demoted regardless of visit count.  Without this
        //      step a high-prior mediocre move (lots of visits, q ≈ 0) can
        //      shadow a sharper line whose q is materially higher, causing
        //      the displayed eval to flicker (e.g. +2.30 dropping to +0.00
        //      when the visit-leader swaps in).
        //   4. Within bucket: visits desc, then q desc.  This keeps the
        //      AlphaZero-canonical visit-based selection among moves that
        //      are roughly equally good in Q.
        constexpr double kProvenWinThreshold = 0.99;
        constexpr double kQBucketDelta       = 0.30;

        // Compute the top Q among non-proven-win candidates so the bucket
        // threshold is independent of the proven-win bump above.
        double max_q_unproven = -2.0;
        for (const auto& c : cands) {
            if (c.q < kProvenWinThreshold) {
                max_q_unproven = std::max(max_q_unproven, c.q);
            }
        }
        const double q_bucket_floor = max_q_unproven - kQBucketDelta;

        std::sort(cands.begin(), cands.end(),
                  [&](const Candidate& a, const Candidate& b) {
            const bool a_pw = a.q >=  kProvenWinThreshold;
            const bool b_pw = b.q >=  kProvenWinThreshold;
            if (a_pw != b_pw) return a_pw;
            if (a_pw) {
                // Both proven wins → play the FASTEST mate.  mate_dist is the
                // ply count along the proven PV (INT_MAX when the win wasn't
                // confirmed terminal within the walk, so confirmed short mates
                // sort ahead of "winning but unproven-distance" lines).  This is
                // what makes the engine actually deliver the mate instead of
                // shuffling around a won position until the 50-move rule.
                if (a.mate_dist != b.mate_dist) return a.mate_dist < b.mate_dist;
                if (a.q != b.q) return a.q > b.q;
                return a.visits > b.visits;
            }
            const bool a_pl = a.q <= -kProvenWinThreshold;
            const bool b_pl = b.q <= -kProvenWinThreshold;
            if (a_pl != b_pl) return !a_pl;  // non-loss first
            if (a_pl) {
                // Both proven losses → drag it out: prefer the LONGEST line so
                // the opponent has the most chances to err.
                if (a.mate_dist != b.mate_dist) return a.mate_dist > b.mate_dist;
                if (a.q != b.q) return a.q > b.q;
                return a.visits > b.visits;
            }
            const bool a_competitive = a.q >= q_bucket_floor;
            const bool b_competitive = b.q >= q_bucket_floor;
            if (a_competitive != b_competitive) return a_competitive;
            if (a.visits != b.visits) return a.visits > b.visits;
            return a.q > b.q;
        });

        r.best_move  = cands.front().move;
        r.root_value = static_cast<float>(cands.front().q);

        const int kmax = std::min<int>(cfg_.multipv, static_cast<int>(cands.size()));
        r.top_pvs.reserve(static_cast<std::size_t>(kmax));
        r.top_values.reserve(static_cast<std::size_t>(kmax));
        r.top_visits.reserve(static_cast<std::size_t>(kmax));
        r.top_proven_sign.reserve(static_cast<std::size_t>(kmax));

        for (int k = 0; k < kmax; ++k) {
            auto& cd = cands[static_cast<std::size_t>(k)];
            r.top_pvs.push_back(std::move(cd.pv));     // precomputed above
            r.top_values.push_back(static_cast<float>(cd.q));
            r.top_visits.push_back(cd.visits);
            r.top_proven_sign.push_back(cd.proven_sign);
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
    //
    // If `out_proven_sign` is non-null and the PV walks into a terminal node,
    // it is set to the terminal value from ROOT'S stm perspective:
    //    +1 = win for root's stm, -1 = loss, 0 = draw / not terminal / unknown.
    // This is the canonical "proven mate" signal callers should use instead
    // of |q| > 0.99, which is racy under concurrent backprop.
    std::vector<Move> build_pv(std::uint32_t start_idx,
                                const Position& root_pos,
                                const Move& first_move,
                                int* out_proven_sign = nullptr) {
        std::vector<Move> pv;
        pv.push_back(first_move);
        Position tmp = root_pos;
        Rules::apply(tmp, first_move);
        // Each applied move flips the side-to-move; track the parity so we can
        // map a leaf's stm-relative terminal value back to root's stm.
        int plies_applied = 1;

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
            ++plies_applied;
            cur = next;
        }

        if (out_proven_sign != nullptr) {
            *out_proven_sign = 0;
            // Terminal at the PV leaf? Two paths set the flag:
            //   * the Node itself was marked terminal during search, or
            //   * the position has no legal moves (catches the case where
            //     the PV ran into a leaf that wasn't expanded yet but is
            //     in fact mate / stalemate).
            const bool node_terminal = nodes_[cur].is_terminal();
            const bool pos_terminal  = node_terminal || Rules::is_terminal(tmp);
            if (pos_terminal) {
                // terminal_stm_value returns value from tmp's stm. Convert to
                // root's stm: odd plies → sign flip, even plies → same sign.
                const float tv = terminal_stm_value(tmp);
                const float root_tv = (plies_applied % 2 == 0) ? tv : -tv;
                if      (root_tv >  0.99f) *out_proven_sign = +1;
                else if (root_tv < -0.99f) *out_proven_sign = -1;
                // |root_tv| ≤ 0.99 (i.e. draw) → leave 0.
            }
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
