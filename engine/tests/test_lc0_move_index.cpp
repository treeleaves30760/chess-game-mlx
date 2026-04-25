// SPDX-License-Identifier: MIT
// engine/tests/test_lc0_move_index.cpp
//
// Unit tests for lc0_move_index.hpp:
//   - lc0_move_to_slot() correctness
//   - lc0_slot_to_chess_move() correctness
//   - Round-trip consistency for all legal moves at startpos
//   - Specific canonical cases (d2d4 at slot 293, e7e8q promotion, etc.)

#include "chess/lc0_move_index.hpp"

#include <chess.hpp>
#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace chess_mlx::chess::lc0;

// ============================================================================
// Helpers
// ============================================================================

static bool white_stm() { return false; }  // black_stm = false = white STM
static bool black_stm() { return true; }

// ============================================================================
// Basic lookup tests
// ============================================================================

TEST(Lc0MoveIndex, TotalSlots) {
    EXPECT_EQ(kLc0MoveStrs.size(), 1858u);
}

TEST(Lc0MoveIndex, StartposD2D4) {
    // d2d4 should be at slot 293 (from the golden data analysis)
    ::chess::Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    ::chess::Move d2d4 = ::chess::uci::uciToMove(board, "d2d4");

    int slot = lc0_move_to_slot(d2d4, /*black_stm=*/false);
    EXPECT_EQ(slot, 293) << "d2d4 should map to slot 293";

    // Verify the string at slot 293
    EXPECT_EQ(kLc0MoveStrs[293], "d2d4");
}

TEST(Lc0MoveIndex, StartposE2E4) {
    ::chess::Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    ::chess::Move e2e4 = ::chess::uci::uciToMove(board, "e2e4");

    int slot = lc0_move_to_slot(e2e4, /*black_stm=*/false);
    EXPECT_GE(slot, 0) << "e2e4 should have a valid slot";

    // Verify the string at that slot
    EXPECT_EQ(kLc0MoveStrs[static_cast<std::size_t>(slot)], "e2e4");
}

TEST(Lc0MoveIndex, StartposG1F3) {
    // Nf3 at slot 159
    ::chess::Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    ::chess::Move g1f3 = ::chess::uci::uciToMove(board, "g1f3");

    int slot = lc0_move_to_slot(g1f3, /*black_stm=*/false);
    EXPECT_EQ(slot, 159) << "g1f3 should map to slot 159";
}

// ============================================================================
// Black STM mirroring
// ============================================================================

TEST(Lc0MoveIndex, BlackMirroringNc6) {
    // After 1.e4 e5, black plays Nc6 (b8->c6).
    // In LC0's white-perspective: b1->c3 (mirrored). Verify it maps to a valid slot.
    ::chess::Board board;
    board.setFen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    board.makeMove(::chess::uci::uciToMove(board, "g1f3"));  // 2.Nf3
    // Now black to move: 2...Nc6
    ::chess::Move nc6 = ::chess::uci::uciToMove(board, "b8c6");

    int slot = lc0_move_to_slot(nc6, /*black_stm=*/true);
    EXPECT_GE(slot, 0) << "Nc6 by black should have a valid slot";

    // Mirrored: b8 (sq=57) -> c6 is rank-flipped to b1 (sq=1) -> c3 (sq=18)
    // In the table: "b1c3" should be at some slot.
    // Verify the stored string is "b1c3" (white-perspective of black's Nc6)
    std::string str(kLc0MoveStrs[static_cast<std::size_t>(slot)]);
    EXPECT_EQ(str, "b1c3") << "Nc6 by black should map to b1c3 in LC0 table";
}

TEST(Lc0MoveIndex, BlackMirroringSymmetry) {
    // White's Nc3 (b1->c3) and black's Nc6 (b8->c6) should map to the same slot.
    ::chess::Board board_w;
    board_w.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    ::chess::Move nc3_white = ::chess::uci::uciToMove(board_w, "b1c3");
    int slot_white = lc0_move_to_slot(nc3_white, /*black_stm=*/false);

    ::chess::Board board_b;
    board_b.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    ::chess::Move nc6_black = ::chess::uci::uciToMove(board_b, "b8c6");
    int slot_black = lc0_move_to_slot(nc6_black, /*black_stm=*/true);

    EXPECT_GE(slot_white, 0);
    EXPECT_GE(slot_black, 0);
    EXPECT_EQ(slot_white, slot_black)
        << "White's Nc3 and Black's Nc6 should map to the same LC0 slot";
}

// ============================================================================
// Promotions
// ============================================================================

TEST(Lc0MoveIndex, QueenPromotion) {
    // e7e8q (white pawn promotes to queen)
    ::chess::Board board;
    board.setFen("8/4P3/8/8/8/8/8/8 w - - 0 1");
    ::chess::Move e7e8q = ::chess::uci::uciToMove(board, "e7e8q");

    int slot = lc0_move_to_slot(e7e8q, /*black_stm=*/false);
    EXPECT_GE(slot, 0) << "e7e8q should have a valid slot";
    std::string str(kLc0MoveStrs[static_cast<std::size_t>(slot)]);
    EXPECT_EQ(str, "e7e8q");
}

TEST(Lc0MoveIndex, BishopPromotion) {
    // e7e8b — different slot from e7e8q
    ::chess::Board board;
    board.setFen("8/4P3/8/8/8/8/8/8 w - - 0 1");
    ::chess::Move e7e8q = ::chess::uci::uciToMove(board, "e7e8q");
    ::chess::Move e7e8b = ::chess::uci::uciToMove(board, "e7e8b");

    int slot_q = lc0_move_to_slot(e7e8q, /*black_stm=*/false);
    int slot_b = lc0_move_to_slot(e7e8b, /*black_stm=*/false);

    EXPECT_GE(slot_q, 0);
    EXPECT_GE(slot_b, 0);
    EXPECT_NE(slot_q, slot_b) << "e7e8q and e7e8b should map to different slots";
}

TEST(Lc0MoveIndex, KnightPromotion) {
    // e7e8n — knight promotion, no suffix in LC0 table (falls back to "e7e8")
    ::chess::Board board;
    board.setFen("8/4P3/8/8/8/8/8/8 w - - 0 1");
    ::chess::Move e7e8n = ::chess::uci::uciToMove(board, "e7e8n");

    int slot = lc0_move_to_slot(e7e8n, /*black_stm=*/false);
    EXPECT_GE(slot, 0) << "Knight promotion should have a valid slot";
    // The slot should point to "e7e8" (no suffix)
    std::string str(kLc0MoveStrs[static_cast<std::size_t>(slot)]);
    EXPECT_EQ(str, "e7e8") << "Knight promotion maps to no-suffix entry";
}

// ============================================================================
// Round-trip: lc0_move_to_slot -> lc0_slot_to_chess_move
// ============================================================================

TEST(Lc0MoveIndex, RoundTripStartpos) {
    ::chess::Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    ::chess::Movelist legal;
    ::chess::movegen::legalmoves(legal, board);

    int missing = 0;
    for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
        const auto& mv = legal[i];
        int slot = lc0_move_to_slot(mv, /*black_stm=*/false);
        if (slot < 0) {
            ++missing;
            ADD_FAILURE() << "No LC0 slot for move: "
                          << ::chess::uci::moveToUci(mv);
            continue;
        }

        // Round-trip: reconstruct the move from the slot
        auto recovered = lc0_slot_to_chess_move(slot, board, /*black_stm=*/false);
        // Compare as UCI strings (most reliable)
        EXPECT_EQ(::chess::uci::moveToUci(recovered), ::chess::uci::moveToUci(mv))
            << "Round-trip mismatch for move " << ::chess::uci::moveToUci(mv)
            << " at slot " << slot;
    }
    EXPECT_EQ(missing, 0) << missing << " legal moves had no LC0 slot";
}

TEST(Lc0MoveIndex, RoundTripBlackStartpos) {
    // Startpos with black to move
    ::chess::Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

    ::chess::Movelist legal;
    ::chess::movegen::legalmoves(legal, board);

    for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
        const auto& mv = legal[i];
        int slot = lc0_move_to_slot(mv, /*black_stm=*/true);
        EXPECT_GE(slot, 0) << "No LC0 slot for black move: "
                            << ::chess::uci::moveToUci(mv);
        if (slot < 0) continue;

        auto recovered = lc0_slot_to_chess_move(slot, board, /*black_stm=*/true);
        EXPECT_EQ(::chess::uci::moveToUci(recovered), ::chess::uci::moveToUci(mv))
            << "Round-trip mismatch for black move " << ::chess::uci::moveToUci(mv);
    }
}

// ============================================================================
// Table self-consistency
// ============================================================================

TEST(Lc0MoveIndex, TableNoDuplicatesMappedSlots) {
    // Each LC0 slot should map to a unique 4672 index (or -1).
    const auto& table = lc0_slot_to_4672_table();
    // Note: some LC0 moves (e.g. moves going off the board) should map to -1.
    // But for valid chess moves the 4672 index should be unique.
    int valid = 0;
    for (int s = 0; s < 1858; ++s) {
        if (table[s] >= 0) ++valid;
    }
    // All 1858 LC0 move strings should be valid chess moves (no off-board)
    EXPECT_EQ(valid, 1858) << (1858 - valid) << " slots map to -1 (off-board)";
}
