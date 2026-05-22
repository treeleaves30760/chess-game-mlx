// SPDX-License-Identifier: MIT
// engine/tests/test_mcts.cpp
//
// MCTS correctness tests using the StubBackend (uniform policy, value 0).

#include <gtest/gtest.h>

#include "chess/chess_traits.hpp"
#include "nn/stub_backend.hpp"
#include "search/mcts.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using chess_mlx::chess::ChessMove;
using chess_mlx::chess::ChessPosition;
using chess_mlx::chess::ChessTraits;
using chess_mlx::search::MCTS;
using chess_mlx::search::TimeControl;
using chess_mlx::nn::StubBackend;

static std::shared_ptr<StubBackend> make_stub() {
    return std::make_shared<StubBackend>(/*policy_size=*/4672, /*input_size=*/64 * 19);
}

// A StubBackend that reports batch_preferred() == true, so MCTS exercises the
// PIPELINED worker_loop (with the Batcher) instead of worker_loop_simple. The
// plain StubBackend leaves batch_preferred() at its false default, so the other
// tests here never touch the pipelined path — which is exactly where the
// "search self-terminates on a resolved mate" bug lived.
class BatchStubBackend : public StubBackend {
public:
    using StubBackend::StubBackend;
    bool batch_preferred() const override { return true; }
};

static std::shared_ptr<BatchStubBackend> make_batch_stub() {
    return std::make_shared<BatchStubBackend>(/*policy_size=*/4672,
                                              /*input_size=*/64 * 19);
}

// ----------------------------------------------------------------------------
// With StubBackend on startpos: search returns a legal move.
// ----------------------------------------------------------------------------
TEST(MCTSChess, StartposReturnsLegalMove) {
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 1;
    cfg.input_size = 64 * 19;
    cfg.policy_size = 4672;
    MCTS<ChessTraits> mcts(backend, cfg);

    TimeControl tc;
    tc.max_nodes = 200;

    ChessPosition pos;
    auto result = mcts.search(pos, tc);

    // Ensure a legal move was returned.
    core::MoveList<ChessMove> ml;
    ChessTraits::generate_legal(pos, ml);
    bool found = false;
    for (const auto& m : ml) {
        if (m == result.best_move) { found = true; break; }
    }
    EXPECT_TRUE(found) << "Best move must be legal from startpos";
    EXPECT_FALSE(result.best_move.is_null());
    EXPECT_GT(result.nodes, 0u);
}

// ----------------------------------------------------------------------------
// Only one legal move: that move must be selected.
// (FEN: white king h8 in mate, only one move available.)
// ----------------------------------------------------------------------------
TEST(MCTSChess, OneLegalMove) {
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 1;
    MCTS<ChessTraits> mcts(backend, cfg);

    // Rook+King mate pattern: black king stuck, white to move has trivial responses.
    // Use a position with exactly one legal move: black king on h8, white queen h7,
    // white king f7; black to move. Only Kxh7 is legal.
    ChessPosition pos("7k/5K1Q/8/8/8/8/8/8 b - - 0 1");
    core::MoveList<ChessMove> ml;
    ChessTraits::generate_legal(pos, ml);
    ASSERT_EQ(ml.size(), 1u) << "Test prerequisite: exactly one legal move";
    const ChessMove only_move = ml[0];

    TimeControl tc;
    tc.max_nodes = 50;
    auto result = mcts.search(pos, tc);

    EXPECT_EQ(result.best_move, only_move);
}

// ----------------------------------------------------------------------------
// Mate-in-1 convergence: given many visits, MCTS should find the mating move.
// Uses a trivial position where one move mates and the rest don't.
// ----------------------------------------------------------------------------
TEST(MCTSChess, MateInOne) {
    // White to play Qh7# or similar.  FEN: K..k with white queen ready to mate.
    // Use: white king g1, white queen g8, black king h6.  White can play Qh8#.
    // Actually let's use a simpler mate-in-1: Fool's mate setup.
    //   rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3
    //   White is in check from Qh4.  Only move to stop check... wait this isn't mate.
    //
    // Use: black king in corner, white queen gives mate on next move.
    //   "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1" — rook back-rank mate setup,
    //   but needs a move.  Let's use: "k7/7R/1K6/8/8/8/8/8 w - - 0 1" — white
    //   plays Rh8#.  Verify mate-in-1.
    ChessPosition pos("k7/7R/1K6/8/8/8/8/8 w - - 0 1");

    // Find the mating move manually.
    core::MoveList<ChessMove> ml;
    ChessTraits::generate_legal(pos, ml);
    ChessMove mating;
    int mates = 0;
    for (const auto& m : ml) {
        ChessPosition tmp = pos;
        ChessTraits::apply(tmp, m);
        // After the move, check if it's checkmate (black has no moves and is in check).
        const auto [reason, result] = tmp.board.isGameOver();
        if (result == ::chess::GameResult::LOSE) {
            // The side to move (black) has lost — so this was mate.
            mates++;
            mating = m;
        }
    }
    ASSERT_GE(mates, 1) << "Test prerequisite: position is mate-in-1";

    // Run MCTS with enough nodes that it should prefer mating.
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 1;
    MCTS<ChessTraits> mcts(backend, cfg);
    TimeControl tc;
    tc.max_nodes = 5000;

    auto result = mcts.search(pos, tc);

    EXPECT_EQ(result.best_move, mating)
        << "MCTS should converge to the mating move given enough visits; got "
        << result.best_move.to_uci() << " expected " << mating.to_uci();
}

// ----------------------------------------------------------------------------
// Multi-thread safety: N concurrent searches, no crash, all return legal.
// ----------------------------------------------------------------------------
TEST(MCTSChess, MultiThreadNoCrash) {
    constexpr int kSearches = 8;
    std::atomic<int> successes{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < kSearches; ++i) {
        threads.emplace_back([&]() {
            auto backend = make_stub();
            MCTS<ChessTraits>::Config cfg;
            cfg.threads = 2;
            MCTS<ChessTraits> mcts(backend, cfg);

            TimeControl tc;
            tc.max_nodes = 100;

            ChessPosition pos;
            auto result = mcts.search(pos, tc);
            if (!result.best_move.is_null()) {
                successes.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) t.join();
    EXPECT_EQ(successes.load(), kSearches);
}

// ----------------------------------------------------------------------------
// Publication race regression for Node::flags.
//
// One writer expands a fresh Node `kIters` times with a monotonically
// increasing pattern in first_child_idx / num_children.  One reader polls
// is_expanded() and, on every observed expansion, must see the matching
// pattern — never a stale 0 from a previous reset.
//
// Ping-pong cadence: writer publishes → reader observes → reader ACKs →
// writer clears → reader waits for clear → loop.  The ACK is a separate
// atomic so we never spin on the SAME atomic we're mutating, which would
// trivially defeat the test on x86's strong memory model.
//
// Pre-fix (plain `flags` with relaxed loads) the reader regularly sees
// is_expanded()=true while first_child_idx / num_children are still 0
// under TSan and on weakly-ordered hardware (arm64).  Post-fix the
// stale_reads counter stays at 0.
// ----------------------------------------------------------------------------
TEST(MCTSChess, NodePublicationVisibility) {
    using chess_mlx::search::Node;

    constexpr int kIters = 10000;

    Node node;
    std::atomic<int> next_pub  {0};   // writer increments after publishing
    std::atomic<int> last_ack  {0};   // reader increments after observing
    std::atomic<int> stale_reads{0};

    std::thread writer([&]() {
        for (int i = 1; i <= kIters; ++i) {
            // Wait for reader to ACK the previous publication.
            while (last_ack.load(std::memory_order_acquire) < i - 1) {
                std::this_thread::yield();
            }
            // Plain writes — must NOT be reorderable past the release-store
            // below, which is the whole point of the protocol.
            node.first_child_idx = static_cast<std::uint32_t>(i);
            node.num_children    = static_cast<std::uint16_t>((i & 0x0FFF) | 0x1000);
            node.set_expanded();              // RELEASE
            next_pub.store(i, std::memory_order_release);

            // Wait until reader saw it before we reset for the next round.
            while (last_ack.load(std::memory_order_acquire) < i) {
                std::this_thread::yield();
            }
            // Clear for next iter — reader is between iterations.
            node.flags.store(0, std::memory_order_relaxed);
            node.first_child_idx = 0;
            node.num_children    = 0;
        }
    });

    std::thread reader([&]() {
        for (int i = 1; i <= kIters; ++i) {
            // Wait for the next publication via the side-channel counter so
            // we don't spin on the same atomic the writer is racing on.
            while (next_pub.load(std::memory_order_acquire) < i) {
                std::this_thread::yield();
            }
            // ACQUIRE on the publication gate; the matching plain reads
            // below must see the writer's monotonic pattern.
            const bool expanded = node.is_expanded();
            const auto fc = node.first_child_idx;
            const auto nc = node.num_children;
            const bool consistent =
                expanded &&
                static_cast<int>(fc) == i &&
                ((nc & 0x0FFF) == (i & 0x0FFF)) &&
                ((nc & 0xF000) == 0x1000);
            if (!consistent) {
                stale_reads.fetch_add(1, std::memory_order_relaxed);
            }
            last_ack.store(i, std::memory_order_release);
        }
    });

    writer.join();
    reader.join();

    EXPECT_EQ(stale_reads.load(), 0)
        << "Observed is_expanded()=true with stale first_child_idx / "
           "num_children — Node::flags publication contract broken.";
}

// ----------------------------------------------------------------------------
// Internal MCTS threading: one search with threads=4 must return a legal move.
// ----------------------------------------------------------------------------
TEST(MCTSChess, InternalMultiThread) {
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 4;
    MCTS<ChessTraits> mcts(backend, cfg);

    TimeControl tc;
    tc.max_nodes = 500;

    ChessPosition pos;
    auto result = mcts.search(pos, tc);

    core::MoveList<ChessMove> ml;
    ChessTraits::generate_legal(pos, ml);
    bool found = false;
    for (const auto& m : ml) if (m == result.best_move) { found = true; break; }
    EXPECT_TRUE(found);
    EXPECT_GT(result.nodes, 0u);
}

// ----------------------------------------------------------------------------
// Regression: the PIPELINED worker_loop must not self-terminate on a position
// whose tree resolves to terminals (a forced mate).  Pre-fix, every worker
// drained its pipeline (select_leaf kept returning terminal leaves that were
// popped synchronously) and exited via `if (pending.empty()) break;`, so a
// search bailed after a few hundred nodes — starving won endgames of the visits
// they need and making `go infinite` finish on its own.  Post-fix the search
// runs until its node budget.
// ----------------------------------------------------------------------------
TEST(MCTSChess, PipelinedSearchUsesFullBudgetOnResolvedMate) {
    auto backend = make_batch_stub();  // batch_preferred() == true → pipelined
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 4;
    cfg.multipv = 3;
    MCTS<ChessTraits> mcts(backend, cfg);

    TimeControl tc;
    tc.max_nodes = 4000;

    // Mate-in-1: the tree near the root resolves to a terminal almost at once.
    ChessPosition pos("k7/7R/1K6/8/8/8/8/8 w - - 0 1");
    auto result = mcts.search(pos, tc);

    EXPECT_GE(result.nodes, 3000u)
        << "pipelined search bailed early on a resolved mate (nodes="
        << result.nodes << ", budget=4000) — worker_loop exited on empty pipeline";
    EXPECT_FALSE(result.best_move.is_null());
}

// ----------------------------------------------------------------------------
// Time control: go movetime 100 terminates within 300ms (generous slack).
// ----------------------------------------------------------------------------
TEST(MCTSChess, MovetimeHonoured) {
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 1;
    MCTS<ChessTraits> mcts(backend, cfg);

    TimeControl tc;
    tc.movetime_ms = 100;

    const auto t0 = std::chrono::steady_clock::now();
    ChessPosition pos;
    auto result = mcts.search(pos, tc);
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    EXPECT_LT(dt, 400) << "movetime 100 should terminate within 400ms";
    EXPECT_FALSE(result.best_move.is_null());
}

// ----------------------------------------------------------------------------
// MultiPV: requesting k=3 should return top_pvs of size 3 (or fewer if
// fewer legal moves than k, which is not the case on startpos).
// ----------------------------------------------------------------------------
TEST(MCTSChess, MultiPVReturnsK) {
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 1;
    cfg.multipv = 3;
    MCTS<ChessTraits> mcts(backend, cfg);

    TimeControl tc;
    tc.max_nodes = 500;

    ChessPosition pos;
    auto result = mcts.search(pos, tc);

    EXPECT_EQ(result.top_pvs.size(), 3u);
    for (auto& pv : result.top_pvs) EXPECT_FALSE(pv.empty());
}

// ----------------------------------------------------------------------------
// Forced-mate signal: on a mate-in-1, the top candidate's PV walks into a
// terminal node and build_result must mark top_proven_sign[0] == +1 (win for
// the side to move).  This is the signal the UCI layer uses to emit `score
// mate N` reliably instead of a flaky |q| > 0.99 threshold check.
// ----------------------------------------------------------------------------
TEST(MCTSChess, MateInOneSetsProvenSign) {
    // White plays Rh8# (rook a1->a8 in some FENs; here h7->h8 from K..k file).
    ChessPosition pos("k7/7R/1K6/8/8/8/8/8 w - - 0 1");

    // Identify the mating move so we can assert the PV starts with it.
    core::MoveList<ChessMove> ml;
    ChessTraits::generate_legal(pos, ml);
    ChessMove mating;
    for (const auto& m : ml) {
        ChessPosition tmp = pos;
        ChessTraits::apply(tmp, m);
        const auto [reason, result] = tmp.board.isGameOver();
        if (result == ::chess::GameResult::LOSE) { mating = m; break; }
    }
    ASSERT_FALSE(mating.is_null()) << "Test prerequisite: position is mate-in-1";

    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads  = 1;
    cfg.multipv  = 3;
    MCTS<ChessTraits> mcts(backend, cfg);
    TimeControl tc;
    tc.max_nodes = 5000;

    auto result = mcts.search(pos, tc);

    ASSERT_FALSE(result.top_pvs.empty());
    ASSERT_FALSE(result.top_proven_sign.empty());
    EXPECT_EQ(result.best_move, mating);
    EXPECT_EQ(result.top_pvs[0].front(), mating);
    EXPECT_EQ(result.top_proven_sign[0], +1)
        << "Mate-in-1 PV should be flagged proven-win for the side to move";
}

// ----------------------------------------------------------------------------
// Non-terminal positions carry no proven flag (sign == 0 for all candidates).
// ----------------------------------------------------------------------------
TEST(MCTSChess, StartposHasNoProvenSign) {
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 1;
    cfg.multipv = 3;
    MCTS<ChessTraits> mcts(backend, cfg);
    TimeControl tc;
    tc.max_nodes = 500;

    ChessPosition pos;
    auto result = mcts.search(pos, tc);

    ASSERT_EQ(result.top_proven_sign.size(), result.top_pvs.size());
    for (int s : result.top_proven_sign) EXPECT_EQ(s, 0);
}

// ----------------------------------------------------------------------------
// Reset behaviour: after reset() the node count drops to 0 or 1 (only root).
// ----------------------------------------------------------------------------
TEST(MCTSChess, ResetClearsTree) {
    auto backend = make_stub();
    MCTS<ChessTraits>::Config cfg;
    cfg.threads = 1;
    MCTS<ChessTraits> mcts(backend, cfg);

    TimeControl tc;
    tc.max_nodes = 100;

    ChessPosition pos;
    mcts.search(pos, tc);
    EXPECT_GT(mcts.node_count(), 1u);

    mcts.reset();
    EXPECT_LE(mcts.node_count(), 1u);
}
