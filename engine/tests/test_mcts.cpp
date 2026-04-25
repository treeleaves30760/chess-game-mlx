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
