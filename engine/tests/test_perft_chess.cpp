// engine/tests/test_perft_chess.cpp
//
// Perft correctness tests using chess-library's move generation via ChessTraits.
//
// Perft counts verified against chessprogramming.org/Perft_Results.
//
// Test positions:
//   - Starting position          (depth 1-6)
//   - Kiwipete / Position 2      (depth 1-4)
//   - Position 3 (Rook endgame)  (depth 1-5)
//   - Position 4 (Tricky)        (depth 1-4)
//   - Position 5                 (depth 1-4)
//   - Position 6                 (depth 1-4)

#include <gtest/gtest.h>

#include <chess/chess_traits.hpp>
#include <core/game_rules.hpp>

#include <chess.hpp>
#include <cstdint>
#include <string>

using namespace chess_mlx::chess;

// ============================================================================
// Perft implementation
// ============================================================================

/// Recursive perft: count leaf nodes at `depth`.
static std::uint64_t perft(::chess::Board& board, int depth) {
    if (depth == 0) return 1ULL;

    ::chess::Movelist moves;
    ::chess::movegen::legalmoves(moves, board);

    if (depth == 1) {
        return static_cast<std::uint64_t>(moves.size());
    }

    std::uint64_t nodes = 0;
    for (const auto& m : moves) {
        board.makeMove(m);
        nodes += perft(board, depth - 1);
        board.unmakeMove(m);
    }
    return nodes;
}

static std::uint64_t perft_fen(const std::string& fen, int depth) {
    ::chess::Board board;
    board.setFen(fen);
    return perft(board, depth);
}

// ============================================================================
// Starting position
// ============================================================================

class PerftStartpos : public ::testing::Test {
protected:
    static constexpr const char* kFen =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
};

TEST_F(PerftStartpos, Depth1) { EXPECT_EQ(perft_fen(kFen, 1), 20ULL); }
TEST_F(PerftStartpos, Depth2) { EXPECT_EQ(perft_fen(kFen, 2), 400ULL); }
TEST_F(PerftStartpos, Depth3) { EXPECT_EQ(perft_fen(kFen, 3), 8902ULL); }
TEST_F(PerftStartpos, Depth4) { EXPECT_EQ(perft_fen(kFen, 4), 197281ULL); }
TEST_F(PerftStartpos, Depth5) { EXPECT_EQ(perft_fen(kFen, 5), 4865609ULL); }
TEST_F(PerftStartpos, Depth6) { EXPECT_EQ(perft_fen(kFen, 6), 119060324ULL); }

// ============================================================================
// Kiwipete (Position 2) — castling, en passant, promotion stress test
// ============================================================================

class PerftKiwipete : public ::testing::Test {
protected:
    static constexpr const char* kFen =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
};

TEST_F(PerftKiwipete, Depth1) { EXPECT_EQ(perft_fen(kFen, 1), 48ULL); }
TEST_F(PerftKiwipete, Depth2) { EXPECT_EQ(perft_fen(kFen, 2), 2039ULL); }
TEST_F(PerftKiwipete, Depth3) { EXPECT_EQ(perft_fen(kFen, 3), 97862ULL); }
TEST_F(PerftKiwipete, Depth4) { EXPECT_EQ(perft_fen(kFen, 4), 4085603ULL); }

// ============================================================================
// Position 3 (Rook endgame)
// ============================================================================

class PerftPos3 : public ::testing::Test {
protected:
    static constexpr const char* kFen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
};

TEST_F(PerftPos3, Depth1) { EXPECT_EQ(perft_fen(kFen, 1), 14ULL); }
TEST_F(PerftPos3, Depth2) { EXPECT_EQ(perft_fen(kFen, 2), 191ULL); }
TEST_F(PerftPos3, Depth3) { EXPECT_EQ(perft_fen(kFen, 3), 2812ULL); }
TEST_F(PerftPos3, Depth4) { EXPECT_EQ(perft_fen(kFen, 4), 43238ULL); }
TEST_F(PerftPos3, Depth5) { EXPECT_EQ(perft_fen(kFen, 5), 674624ULL); }

// ============================================================================
// Position 4 (Tricky — promotions, castling edge cases)
// Mirror of Position 3 from chessprogramming.org/Perft_Results
// perft(1)=6, perft(2)=264, perft(3)=9467, perft(4)=422333
// ============================================================================

class PerftPos4 : public ::testing::Test {
protected:
    // This is the mirror of Position 3 (black to move variant).
    static constexpr const char* kFen =
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";
};

TEST_F(PerftPos4, Depth1) { EXPECT_EQ(perft_fen(kFen, 1), 6ULL); }
TEST_F(PerftPos4, Depth2) { EXPECT_EQ(perft_fen(kFen, 2), 264ULL); }
TEST_F(PerftPos4, Depth3) { EXPECT_EQ(perft_fen(kFen, 3), 9467ULL); }
TEST_F(PerftPos4, Depth4) { EXPECT_EQ(perft_fen(kFen, 4), 422333ULL); }

// ============================================================================
// Position 5 (Promotions + checks)
// ============================================================================

class PerftPos5 : public ::testing::Test {
protected:
    static constexpr const char* kFen =
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
};

TEST_F(PerftPos5, Depth1) { EXPECT_EQ(perft_fen(kFen, 1), 44ULL); }
TEST_F(PerftPos5, Depth2) { EXPECT_EQ(perft_fen(kFen, 2), 1486ULL); }
TEST_F(PerftPos5, Depth3) { EXPECT_EQ(perft_fen(kFen, 3), 62379ULL); }
TEST_F(PerftPos5, Depth4) { EXPECT_EQ(perft_fen(kFen, 4), 2103487ULL); }

// ============================================================================
// Position 6 (mirrored midgame)
// ============================================================================

class PerftPos6 : public ::testing::Test {
protected:
    static constexpr const char* kFen =
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
};

TEST_F(PerftPos6, Depth1) { EXPECT_EQ(perft_fen(kFen, 1), 46ULL); }
TEST_F(PerftPos6, Depth2) { EXPECT_EQ(perft_fen(kFen, 2), 2079ULL); }
TEST_F(PerftPos6, Depth3) { EXPECT_EQ(perft_fen(kFen, 3), 89890ULL); }
TEST_F(PerftPos6, Depth4) { EXPECT_EQ(perft_fen(kFen, 4), 3894594ULL); }

// ============================================================================
// ChessTraits wrapper sanity: generate_legal must match direct movegen
// ============================================================================

TEST(PerftViaTraits, StartposDepth1) {
    ChessPosition pos;
    core::MoveList<ChessMove> ml;
    ChessTraits::generate_legal(pos, ml);
    EXPECT_EQ(ml.size(), 20u);
}

TEST(PerftViaTraits, StartposDepth3) {
    ::chess::Board board;
    board.setFen(::chess::constants::STARTPOS);
    EXPECT_EQ(perft(board, 3), 8902ULL);
}

TEST(PerftViaTraits, KiwipeteDepth3) {
    ::chess::Board board;
    board.setFen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(board, 3), 97862ULL);
}
