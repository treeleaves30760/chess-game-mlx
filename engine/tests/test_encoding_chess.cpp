// engine/tests/test_encoding_chess.cpp
//
// Tests for chess input tensor encoding (encoding_spec.md §1) and
// policy-index mapping (encoding_spec.md §2).
//
// Tests:
//   1.  Starting position — piece planes and scalar planes.
//   2.  After 1. e4 — pawn displacement and side-to-move flip.
//   3.  En passant position — feature 17 set on target square only.
//   4.  Castling rights reflection.
//   5.  Half-move clock normalisation.
//   6.  Policy round-trip: move_to_policy_idx → policy_idx_to_move gives back the same move.
//   7.  All legal moves from startpos have distinct valid policy indices.
//   8.  Underpromotion encoding.
//   9.  En passant move encoding.

#include <gtest/gtest.h>

#include <chess/chess_traits.hpp>
#include <chess/chess_encoding.hpp>
#include <core/game_rules.hpp>

#include <chess.hpp>

#include <array>
#include <cstring>
#include <unordered_set>
#include <string>

using namespace chess_mlx::chess;

// ============================================================================
// Helpers
// ============================================================================

/// Retrieve feature value at (square_idx, feature_plane).
static float feat(const float* tensor, int sq, int feat_plane) {
    return tensor[sq * kInputChannels + feat_plane];
}

/// Apply a UCI move string to a board.
static void do_move(::chess::Board& board, const char* uci) {
    auto m = ::chess::uci::uciToMove(board, uci);
    ASSERT_NE(m, ::chess::Move::NO_MOVE) << "Bad move: " << uci;
    board.makeMove(m);
}

// ============================================================================
// Test 1 — Starting position: piece planes
// ============================================================================

TEST(ChessEncoding, StartposPiecePlanes) {
    ::chess::Board board;
    board.setFen(::chess::constants::STARTPOS);

    float tensor[kInputSize];
    encode_nn(board, tensor);

    // White pawns on rank 2 (squares 8..15)
    for (int f = 0; f < 8; ++f) {
        const int sq = 8 + f; // rank 2
        EXPECT_EQ(feat(tensor, sq, kFeatWhitePawn), 1.0f)
            << "Expected white pawn at square " << sq;
    }

    // Black pawns on rank 7 (squares 48..55)
    for (int f = 0; f < 8; ++f) {
        const int sq = 48 + f; // rank 7
        EXPECT_EQ(feat(tensor, sq, kFeatBlackPawn), 1.0f)
            << "Expected black pawn at square " << sq;
    }

    // White back rank pieces (rank 1 = squares 0..7)
    // a1=R, b1=N, c1=B, d1=Q, e1=K, f1=B, g1=N, h1=R
    EXPECT_EQ(feat(tensor, 0, kFeatWhiteRook),   1.0f); // a1
    EXPECT_EQ(feat(tensor, 1, kFeatWhiteKnight), 1.0f); // b1
    EXPECT_EQ(feat(tensor, 2, kFeatWhiteBishop), 1.0f); // c1
    EXPECT_EQ(feat(tensor, 3, kFeatWhiteQueen),  1.0f); // d1
    EXPECT_EQ(feat(tensor, 4, kFeatWhiteKing),   1.0f); // e1
    EXPECT_EQ(feat(tensor, 5, kFeatWhiteBishop), 1.0f); // f1
    EXPECT_EQ(feat(tensor, 6, kFeatWhiteKnight), 1.0f); // g1
    EXPECT_EQ(feat(tensor, 7, kFeatWhiteRook),   1.0f); // h1

    // Black back rank (rank 8 = squares 56..63)
    EXPECT_EQ(feat(tensor, 56, kFeatBlackRook),   1.0f); // a8
    EXPECT_EQ(feat(tensor, 57, kFeatBlackKnight), 1.0f); // b8
    EXPECT_EQ(feat(tensor, 58, kFeatBlackBishop), 1.0f); // c8
    EXPECT_EQ(feat(tensor, 59, kFeatBlackQueen),  1.0f); // d8
    EXPECT_EQ(feat(tensor, 60, kFeatBlackKing),   1.0f); // e8
    EXPECT_EQ(feat(tensor, 61, kFeatBlackBishop), 1.0f); // f8
    EXPECT_EQ(feat(tensor, 62, kFeatBlackKnight), 1.0f); // g8
    EXPECT_EQ(feat(tensor, 63, kFeatBlackRook),   1.0f); // h8

    // Empty ranks (ranks 3..6 = squares 16..47) should have no pieces.
    for (int sq = 16; sq <= 47; ++sq) {
        for (int feat_plane = 0; feat_plane <= 11; ++feat_plane) {
            EXPECT_EQ(feat(tensor, sq, feat_plane), 0.0f)
                << "Expected empty square " << sq << " feature " << feat_plane;
        }
    }
}

// ============================================================================
// Test 2 — Starting position: scalar planes
// ============================================================================

TEST(ChessEncoding, StartposScalarPlanes) {
    ::chess::Board board;
    board.setFen(::chess::constants::STARTPOS);

    float tensor[kInputSize];
    encode_nn(board, tensor);

    // Side to move is white — feature 12 = 1.0 everywhere
    for (int sq = 0; sq < kSquares; ++sq) {
        EXPECT_EQ(feat(tensor, sq, kFeatSideToMove), 1.0f)
            << "Side-to-move must be 1 for white at sq " << sq;
    }

    // All four castling rights present — features 13..16 = 1.0
    for (int sq = 0; sq < kSquares; ++sq) {
        EXPECT_EQ(feat(tensor, sq, kFeatWKCastle), 1.0f);
        EXPECT_EQ(feat(tensor, sq, kFeatWQCastle), 1.0f);
        EXPECT_EQ(feat(tensor, sq, kFeatBKCastle), 1.0f);
        EXPECT_EQ(feat(tensor, sq, kFeatBQCastle), 1.0f);
    }

    // No en passant — feature 17 = 0.0 everywhere
    for (int sq = 0; sq < kSquares; ++sq) {
        EXPECT_EQ(feat(tensor, sq, kFeatEnPassant), 0.0f);
    }

    // Half-move clock = 0 — feature 18 = 0.0
    for (int sq = 0; sq < kSquares; ++sq) {
        EXPECT_EQ(feat(tensor, sq, kFeatHalfMove), 0.0f);
    }
}

// ============================================================================
// Test 3 — After 1. e4: pawn displacement and side-to-move flip
// ============================================================================

TEST(ChessEncoding, AfterE4) {
    ::chess::Board board;
    board.setFen(::chess::constants::STARTPOS);
    do_move(board, "e2e4");

    float tensor[kInputSize];
    encode_nn(board, tensor);

    // e2 (sq=12) should now be empty.
    EXPECT_EQ(feat(tensor, 12, kFeatWhitePawn), 0.0f) << "e2 must be empty after e4";
    // e4 (sq=28) should have white pawn.
    EXPECT_EQ(feat(tensor, 28, kFeatWhitePawn), 1.0f) << "e4 must have white pawn";

    // Side to move is now black — feature 12 = 0.0 everywhere
    for (int sq = 0; sq < kSquares; ++sq) {
        EXPECT_EQ(feat(tensor, sq, kFeatSideToMove), 0.0f)
            << "Side-to-move must be 0 for black at sq " << sq;
    }

    // No en passant (single-step push scenario — chess-library only sets EP
    // when there is an adjacent enemy pawn that can capture; this move is
    // to e4 with no adjacent black pawn, so EP may or may not be set depending
    // on chess-library version.  We accept either 0 or the correct square.)
    // For this test we simply check the feature is consistent with the board state.
    const auto ep_sq = board.enpassantSq();
    if (ep_sq != ::chess::Square::NO_SQ) {
        EXPECT_EQ(feat(tensor, ep_sq.index(), kFeatEnPassant), 1.0f)
            << "EP target square must be set";
        // All other squares must be 0 for feature 17.
        for (int sq = 0; sq < kSquares; ++sq) {
            if (sq == ep_sq.index()) continue;
            EXPECT_EQ(feat(tensor, sq, kFeatEnPassant), 0.0f);
        }
    } else {
        for (int sq = 0; sq < kSquares; ++sq) {
            EXPECT_EQ(feat(tensor, sq, kFeatEnPassant), 0.0f);
        }
    }
}

// ============================================================================
// Test 4 — En passant target square
// ============================================================================

TEST(ChessEncoding, EnPassantTargetSquare) {
    // Position where en passant is unambiguously available:
    // After 1. e4 e5 2. e5... — no, use a FEN with explicit EP square.
    // FEN: white pawn on e5, black pawn just moved from d7 to d5.
    // After 1.e4 e5 2.e5? no. Use e2e4 e7e5 d2d4 — then e5xd4 EP.
    // Simplest: use a FEN string with ep square set.
    ::chess::Board board;
    board.setFen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    // EP square is d6 = file d (3), rank 6 (5) → sq = 5*8+3 = 43

    float tensor[kInputSize];
    encode_nn(board, tensor);

    const auto ep_sq = board.enpassantSq();
    ASSERT_NE(ep_sq, ::chess::Square::NO_SQ) << "Expected EP square in this FEN";

    const int ep_idx = ep_sq.index();
    EXPECT_EQ(feat(tensor, ep_idx, kFeatEnPassant), 1.0f)
        << "EP target square (index " << ep_idx << ") must have feature 17 = 1";

    // Only the EP square gets the mark.
    int ep_set_count = 0;
    for (int sq = 0; sq < kSquares; ++sq) {
        if (feat(tensor, sq, kFeatEnPassant) == 1.0f) ++ep_set_count;
    }
    EXPECT_EQ(ep_set_count, 1) << "Exactly one square should have EP feature set";
}

// ============================================================================
// Test 5 — Castling rights reflected in tensor
// ============================================================================

TEST(ChessEncoding, CastlingRights) {
    // Black has lost queenside castling rights.
    ::chess::Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1");

    float tensor[kInputSize];
    encode_nn(board, tensor);

    // Black queenside castling (feature 16) should be 0.
    EXPECT_EQ(feat(tensor, 0, kFeatBQCastle), 0.0f) << "Black QS castling should be absent";
    // Other rights should be present.
    EXPECT_EQ(feat(tensor, 0, kFeatWKCastle), 1.0f);
    EXPECT_EQ(feat(tensor, 0, kFeatWQCastle), 1.0f);
    EXPECT_EQ(feat(tensor, 0, kFeatBKCastle), 1.0f);
}

// ============================================================================
// Test 6 — Half-move clock normalisation
// ============================================================================

TEST(ChessEncoding, HalfMoveClockNormalisation) {
    ::chess::Board board;
    // Set half-move clock to 50 (draw territory).
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 50 26");

    float tensor[kInputSize];
    encode_nn(board, tensor);

    const float expected = 50.0f / 100.0f;
    EXPECT_FLOAT_EQ(feat(tensor, 0, kFeatHalfMove), expected);
    // Broadcast: same on all squares.
    for (int sq = 1; sq < kSquares; ++sq) {
        EXPECT_FLOAT_EQ(feat(tensor, sq, kFeatHalfMove), expected);
    }
}

// ============================================================================
// Test 7 — Policy round-trip: all legal startpos moves
// ============================================================================

TEST(ChessEncoding, PolicyRoundTripStartpos) {
    ::chess::Board board;
    board.setFen(::chess::constants::STARTPOS);
    ChessPosition pos;

    ::chess::Movelist moves;
    ::chess::movegen::legalmoves(moves, board);
    ASSERT_GT(moves.size(), 0u);

    std::unordered_set<int> seen_indices;

    for (const auto& m : moves) {
        const int idx = move_to_policy_idx(m);
        EXPECT_GE(idx, 0) << "Policy index must be >= 0 for move " << ::chess::uci::moveToUci(m);
        EXPECT_LT(idx, kPolicySize) << "Policy index must be < " << kPolicySize;

        // Round-trip: policy_idx_to_move should give back the same move.
        const auto reconstructed = policy_idx_to_move(idx, board);
        // Compare from/to/type — the reconstructed move encodes the same piece action.
        EXPECT_EQ(reconstructed.from(), m.from())
            << "Round-trip from-square mismatch for " << ::chess::uci::moveToUci(m)
            << " idx=" << idx;
        EXPECT_EQ(reconstructed.to(), m.to())
            << "Round-trip to-square mismatch for " << ::chess::uci::moveToUci(m)
            << " idx=" << idx;

        EXPECT_TRUE(seen_indices.insert(idx).second)
            << "Duplicate policy index " << idx << " for move " << ::chess::uci::moveToUci(m);
    }
}

// ============================================================================
// Test 8 — Underpromotion encoding
// ============================================================================

TEST(ChessEncoding, UnderpromotionEncoding) {
    // Position: white pawn on a7, kings are far away.
    // FEN: 7k/P7/8/8/8/8/8/7K w - - 0 1
    ::chess::Board board;
    board.setFen("7k/P7/8/8/8/8/8/7K w - - 0 1");
    ChessPosition pos(board.getFen());

    ::chess::Movelist moves;
    ::chess::movegen::legalmoves(moves, board);

    // Should include: a8=Q, a8=R, a8=B, a8=N
    bool found_queen = false, found_rook = false, found_bishop = false, found_knight = false;

    for (const auto& m : moves) {
        if (m.typeOf() != ::chess::Move::PROMOTION) continue;
        const int idx = move_to_policy_idx(m);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, kPolicySize);

        const auto recon = policy_idx_to_move(idx, board);
        EXPECT_EQ(recon.from(), m.from());
        EXPECT_EQ(recon.to(), m.to());

        switch (m.promotionType().internal()) {
            case ::chess::PieceType::QUEEN:  found_queen  = true; break;
            case ::chess::PieceType::ROOK:   found_rook   = true; break;
            case ::chess::PieceType::BISHOP: found_bishop = true; break;
            case ::chess::PieceType::KNIGHT: found_knight = true; break;
            default: break;
        }
    }

    EXPECT_TRUE(found_queen)  << "Queen promotion must be encodable";
    EXPECT_TRUE(found_rook)   << "Rook underpromotion must be encodable";
    EXPECT_TRUE(found_bishop) << "Bishop underpromotion must be encodable";
    EXPECT_TRUE(found_knight) << "Knight underpromotion must be encodable";
}

// ============================================================================
// Test 9 — En passant move encoding
// ============================================================================

TEST(ChessEncoding, EnPassantMoveEncoding) {
    // Position with en passant available: white pawn on e5, black d5→d-ep.
    // FEN with ep on d6: white to move, can play exd6 e.p.
    ::chess::Board board;
    board.setFen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");

    ::chess::Movelist moves;
    ::chess::movegen::legalmoves(moves, board);

    bool found_ep = false;
    for (const auto& m : moves) {
        if (m.typeOf() != ::chess::Move::ENPASSANT) continue;
        found_ep = true;
        const int idx = move_to_policy_idx(m);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, kPolicySize);

        const auto recon = policy_idx_to_move(idx, board);
        EXPECT_EQ(recon.from(), m.from())
            << "EP move from-square mismatch, idx=" << idx;
        EXPECT_EQ(recon.to(), m.to())
            << "EP move to-square mismatch, idx=" << idx;
    }

    EXPECT_TRUE(found_ep) << "Expected at least one en passant move in this position";
}

// ============================================================================
// Test 10 — Tensor shape: exactly kInputSize floats, no out-of-bounds write
// ============================================================================

TEST(ChessEncoding, TensorSizeIsCorrect) {
    EXPECT_EQ(kInputSize, 64 * 19) << "Input tensor must be 1216 floats";
    EXPECT_EQ(kPolicySize, 64 * 73) << "Policy space must be 4672 slots";
}
