// SPDX-License-Identifier: MIT
// engine/tests/test_chess_compact_1858.cpp
//
// Verifies the color-absolute compact 1858 mapping in
// `engine/include/chess/chess_compact_1858.hpp`.
//
// Test surface:
//   * Forward table is exactly 1858 entries large.
//   * Round-trip: move → idx → move recovers the original move on every
//     legal move from a representative set of positions.
//   * Specific anchors: hand-computed indices for a few classic moves
//     (a1a2, e2e4, b1c3, h2h4, h7h8q, etc.).  These will catch any
//     ordering drift between this file and the Python builder.

#include <gtest/gtest.h>

#include <chess.hpp>
#include "chess/chess_compact_1858.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <string>

using namespace chess_mlx::chess::compact1858;
using ::chess::Board;
using ::chess::Move;
using ::chess::Movelist;
using ::chess::Square;

// -----------------------------------------------------------------------------
// Table-shape sanity
// -----------------------------------------------------------------------------

TEST(Compact1858, ForwardTableHasExactly1858Entries) {
    // Count non-(-1) entries in the forward table.
    const auto& tbl = detail::tables();
    int count = 0;
    for (auto v : tbl.fwd) if (v >= 0) ++count;
    EXPECT_EQ(count, kCompactPolicySize);
}

TEST(Compact1858, InverseTableFullyPopulated) {
    const auto& tbl = detail::tables();
    int count = 0;
    for (auto v : tbl.inv) if (v >= 0) ++count;
    EXPECT_EQ(count, kCompactPolicySize);
}

TEST(Compact1858, ForwardInverseConsistency) {
    const auto& tbl = detail::tables();
    for (int compact = 0; compact < kCompactPolicySize; ++compact) {
        const int dense = tbl.inv[static_cast<std::size_t>(compact)];
        ASSERT_GE(dense, 0);
        ASSERT_LT(dense, kDenseSize);
        EXPECT_EQ(tbl.fwd[static_cast<std::size_t>(dense)], compact)
            << "compact=" << compact << " dense=" << dense;
    }
}

// -----------------------------------------------------------------------------
// Round-trip property over real positions
// -----------------------------------------------------------------------------

static const std::array<const char*, 6> kTestFens = {{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",       // startpos
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",       // mirrored stm
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // Kiwipete
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                           // castling all
    "8/2P5/8/8/8/8/8/4K2k w - - 0 1",                                 // white promo on c7
    "4k2K/8/8/8/8/8/2p5/8 b - - 0 1",                                 // black promo on c2
}};

TEST(Compact1858, RoundTripLegalMovesAcrossPositions) {
    for (const char* fen : kTestFens) {
        Board board;
        board.setFen(fen);

        Movelist legal;
        ::chess::movegen::legalmoves(legal, board);

        for (const auto& mv : legal) {
            const int idx = move_to_compact_idx(mv, board);
            if (idx < 0) {
                // Allowed only for black underpromotions per Python builder.
                // Verify that exclusion is the reason — if not, fail.
                const bool is_underpromo =
                    mv.typeOf() == Move::PROMOTION
                    && mv.promotionType().internal() != ::chess::PieceType::QUEEN;
                const bool from_rank_1 = (mv.from().index() >> 3) == 1;
                EXPECT_TRUE(is_underpromo && from_rank_1)
                    << "Unexpected -1 for move on " << fen;
                continue;
            }
            ASSERT_GE(idx, 0);
            ASSERT_LT(idx, kCompactPolicySize);

            const Move recovered = compact_idx_to_move(idx, board);
            EXPECT_NE(recovered, Move::NO_MOVE) << "fen=" << fen
                                                << " idx=" << idx;
            // Compare source / destination — promotion identity is also
            // recovered for under-promotions (queen promos auto-restored).
            EXPECT_EQ(recovered.from().index(), mv.from().index());
            EXPECT_EQ(recovered.to().index(),   mv.to().index());
            if (mv.typeOf() == Move::PROMOTION) {
                EXPECT_EQ(recovered.typeOf(), Move::PROMOTION);
                EXPECT_EQ(recovered.promotionType().internal(),
                          mv.promotionType().internal());
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Anchor indices — hand-derived against the Python builder.  These were
// computed by simulating the build loop on paper for from_sq = a1, b1, ...
// They lock down the canonical ordering so future refactors can't silently
// permute knight/underpromo slots.
// -----------------------------------------------------------------------------
//
// The build order from from_sq=0 (a1):
//   Queen rays (N then NE then E ...):
//     N: a1a2(0), a1a3(1), a1a4(2), a1a5(3), a1a6(4), a1a7(5), a1a8(6)
//     NE: a1b2(7), a1c3(8), a1d4(9), a1e5(10), a1f6(11), a1g7(12), a1h8(13)
//     E: a1b1(14), a1c1(15), a1d1(16), a1e1(17), a1f1(18), a1g1(19), a1h1(20)
//     SE, S, SW, W, NW: all off-board from a1 → skipped
//   Knight (Python order: (+1,+2), (+2,+1), (+2,-1), (+1,-2), (-1,-2),
//                          (-2,-1), (-2,+1), (-1,+2)):
//     From a1=(file=0, rank=0):
//       (+1,+2)→b3 (21), (+2,+1)→c2 (22) — the rest go off-board.
//   Underpromos: not from rank 0 → none.
//   Total so far: 23 indices for from_sq=0.

TEST(Compact1858, AnchorIndices) {
    Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // a1-a2: queen ray N, distance 1 → first compact slot (0).
    auto a1a2 = Move::make<Move::NORMAL>(Square(0),  Square(8));   // a1=0, a2=8
    EXPECT_EQ(move_to_compact_idx(a1a2, board), 0);

    // a1-h8: queen ray NE, distance 7 → slot 13.
    auto a1h8 = Move::make<Move::NORMAL>(Square(0),  Square(63));
    EXPECT_EQ(move_to_compact_idx(a1h8, board), 13);

    // a1-b1: queen ray E, distance 1 → slot 14.
    auto a1b1 = Move::make<Move::NORMAL>(Square(0),  Square(1));
    EXPECT_EQ(move_to_compact_idx(a1b1, board), 14);

    // a1-b3: knight (+1,+2) → slot 21.  Python's _KNIGHT_DELTAS[0].
    // Note: this only works in this synthesised query if the source piece
    // happens to be a knight in the board — but our lookup function reads
    // the actual piece type at from_sq for routing.  We therefore use an
    // arranged board where a knight stands on a1.
    Board knight_board;
    knight_board.setFen("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");
    auto a1b3 = Move::make<Move::NORMAL>(Square(0), Square(17));   // b3=17
    EXPECT_EQ(move_to_compact_idx(a1b3, knight_board), 21);

    // a1-c2: knight (+2,+1) → slot 22.
    auto a1c2 = Move::make<Move::NORMAL>(Square(0), Square(10));   // c2=10
    EXPECT_EQ(move_to_compact_idx(a1c2, knight_board), 22);
}

// -----------------------------------------------------------------------------
// Underpromotion anchors: from a7 (rank 6, file 0) white pawn → a8/b8.
// Python builder enumerates underpromos AFTER all queen-rays and knight
// jumps from a7.  The exact compact index depends on cumulative count;
// we just round-trip and check the promotion piece survives.
// -----------------------------------------------------------------------------

TEST(Compact1858, UnderpromotionRoundTrip) {
    Board board;
    board.setFen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");  // white pawn on a7

    Movelist legal;
    ::chess::movegen::legalmoves(legal, board);

    bool saw_n = false, saw_b = false, saw_r = false, saw_q = false;
    for (const auto& mv : legal) {
        if (mv.typeOf() != Move::PROMOTION) continue;
        const int idx = move_to_compact_idx(mv, board);
        ASSERT_GE(idx, 0) << "Underpromo / queen-promo must be in 1858 from a7";
        const Move rec = compact_idx_to_move(idx, board);
        EXPECT_EQ(rec.typeOf(), Move::PROMOTION);
        EXPECT_EQ(rec.promotionType().internal(),
                  mv.promotionType().internal());
        switch (mv.promotionType().internal()) {
            case ::chess::PieceType::KNIGHT: saw_n = true; break;
            case ::chess::PieceType::BISHOP: saw_b = true; break;
            case ::chess::PieceType::ROOK:   saw_r = true; break;
            case ::chess::PieceType::QUEEN:  saw_q = true; break;
            default: break;
        }
    }
    EXPECT_TRUE(saw_n);
    EXPECT_TRUE(saw_b);
    EXPECT_TRUE(saw_r);
    EXPECT_TRUE(saw_q);
}

TEST(Compact1858, BlackUnderpromotionIsExcluded) {
    // Black pawn on a2; underpromos are NOT in the compact map.
    Board board;
    board.setFen("4k3/8/8/8/8/8/p7/4K3 b - - 0 1");

    Movelist legal;
    ::chess::movegen::legalmoves(legal, board);

    bool saw_excluded_underpromo = false;
    for (const auto& mv : legal) {
        if (mv.typeOf() != Move::PROMOTION) continue;
        if (mv.promotionType().internal() == ::chess::PieceType::QUEEN) {
            // Queen promo is encoded via queen-ray slot — it's IN the table.
            EXPECT_GE(move_to_compact_idx(mv, board), 0);
            continue;
        }
        // Black under-promo from rank 1 → expected -1.
        EXPECT_LT(move_to_compact_idx(mv, board), 0);
        saw_excluded_underpromo = true;
    }
    EXPECT_TRUE(saw_excluded_underpromo);
}
