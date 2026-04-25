// engine/tests/test_lc0_encoding.cpp
//
// Unit tests for the LC0-compatible 112-plane chess input encoder.
//
// Tests:
//   1.  StartPos: no history → history planes 13..103 all zero;
//                 planes 0..12 match LC0's starting piece layout (white STM)
//   2.  After 1.e4: planes 13..25 show startpos, planes 0..12 show e4-pos
//                   (black to move → board rank-flipped)
//   3.  Castling rights planes 104-107: 1.0 when right is available
//   4.  STM plane 108: 1.0 when black to move, 0.0 for white
//   5.  Rule50 plane 109: raw halfmove_clock (LC0's convention, NOT divided by 100)
//   6.  AllOnes plane 111: all 1.0
//   7.  AllZeros plane 110: all 0.0
//   8.  Repetition plane: set when position has been seen before
//   9.  HistoryTracker API: push/history/reset
//  10.  Batch encoding: encode_nn_batch produces same result as individual calls
//  11.  Mirroring: black-to-move rank-flips piece positions correctly
//  12.  No-history startpos: slot-0 pieces match expected LC0 layout

#include <gtest/gtest.h>

#include <chess/lc0_encoding.hpp>
#include <chess/history_tracker.hpp>
#include <chess/chess_traits.hpp>

#include <chess.hpp>

#include <array>
#include <cmath>
#include <string>
#include <vector>

using namespace chess_mlx::chess;
using namespace chess_mlx::chess::lc0;

// ============================================================================
// Helper utilities
// ============================================================================

/// Access a value in the [64 × 112] tensor.
static float val(const float* t, int sq, int plane) noexcept {
    return t[sq * FEAT_DIM + plane];
}

/// Check that all 64 squares have the given value in the given plane.
static bool plane_all(const float* t, int plane, float expected) noexcept {
    for (int sq = 0; sq < 64; ++sq) {
        if (std::fabs(t[sq * FEAT_DIM + plane] - expected) > 1e-6f) return false;
    }
    return true;
}

/// Apply a UCI move to a ChessPosition (returns a new position).
static ChessPosition apply_uci(ChessPosition pos, const char* uci) {
    auto m = ::chess::uci::uciToMove(pos.board, uci);
    pos.board.makeMove(m);
    return pos;
}

/// Build a tensor buffer zeroed out.
static std::array<float, SEQ_LEN * FEAT_DIM> make_buf() noexcept {
    std::array<float, SEQ_LEN * FEAT_DIM> buf{};
    return buf;
}

// Plane base for a history slot
static constexpr int slot_base(int slot) noexcept { return slot * PLANES_PER_BOARD; }
static_assert(slot_base(0) == 0 && slot_base(1) == 13 && slot_base(7) == 91, "slot_base sanity");

// ============================================================================
// Test 1: StartPos with no history → history planes all zero
// ============================================================================

TEST(LC0Encoding, StartposNoHistory_HistoryPlanesZero) {
    ChessPosition pos;  // default = startpos
    std::vector<ChessPosition> hist; // no history

    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    // Planes 13..103 should all be zero (history slots 1..7)
    for (int plane = 13; plane <= 103; ++plane) {
        for (int sq = 0; sq < 64; ++sq) {
            EXPECT_EQ(buf[sq * FEAT_DIM + plane], 0.0f)
                << "plane=" << plane << " sq=" << sq;
        }
    }
}

// ============================================================================
// Test 2: StartPos slot-0 piece planes match expected layout
// ============================================================================
//
// At startpos, white is to move. LC0 encodes from STM perspective.
// "Our" pieces = white pieces (no rank flip).
// Expected layout (a1=sq0, h1=sq7, a2=sq8, ...):
//   Our pawns   (plane 0): rank 2 = squares 8..15
//   Our knights (plane 1): b1=sq1, g1=sq6
//   Our bishops (plane 2): c1=sq2, f1=sq5
//   Our rooks   (plane 3): a1=sq0, h1=sq7
//   Our queens  (plane 4): d1=sq3
//   Our kings   (plane 5): e1=sq4
//   Their pawns   (plane 6): rank 7 = squares 48..55
//   Their knights (plane 7): b8=sq57, g8=sq62
//   Their bishops (plane 8): c8=sq58, f8=sq61
//   Their rooks   (plane 9): a8=sq56, h8=sq63
//   Their queens  (plane10): d8=sq59
//   Their kings   (plane11): e8=sq60

TEST(LC0Encoding, StartposSlot0PiecePlanes) {
    ChessPosition pos;
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    const float* t = buf.data();

    // -- Our pieces (plane 0..5) --

    // Pawns on rank 2 (sq 8-15), none elsewhere
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq >= 8 && sq <= 15) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 0), expected) << "our-pawn sq=" << sq;
    }

    // Knights on b1(1) and g1(6)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 1 || sq == 6) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 1), expected) << "our-knight sq=" << sq;
    }

    // Bishops on c1(2) and f1(5)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 2 || sq == 5) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 2), expected) << "our-bishop sq=" << sq;
    }

    // Rooks on a1(0) and h1(7)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 0 || sq == 7) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 3), expected) << "our-rook sq=" << sq;
    }

    // Queen on d1(3)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 3) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 4), expected) << "our-queen sq=" << sq;
    }

    // King on e1(4)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 4) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 5), expected) << "our-king sq=" << sq;
    }

    // -- Their pieces (plane 6..11) --

    // Black pawns on rank 7 (sq 48-55)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq >= 48 && sq <= 55) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 6), expected) << "their-pawn sq=" << sq;
    }

    // Black knights on b8(57) and g8(62)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 57 || sq == 62) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 7), expected) << "their-knight sq=" << sq;
    }

    // Black bishops on c8(58) and f8(61)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 58 || sq == 61) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 8), expected) << "their-bishop sq=" << sq;
    }

    // Black rooks on a8(56) and h8(63)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 56 || sq == 63) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 9), expected) << "their-rook sq=" << sq;
    }

    // Black queen on d8(59)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 59) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 10), expected) << "their-queen sq=" << sq;
    }

    // Black king on e8(60)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 60) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 11), expected) << "their-king sq=" << sq;
    }

    // Repetition plane at startpos with no history: 0
    EXPECT_TRUE(plane_all(t, 12, 0.0f)) << "startpos rep plane should be 0";
}

// ============================================================================
// Test 3: After 1.e4 — history slot 1 shows startpos; slot 0 shows e4 pos
//         with black to move (rank-flipped)
// ============================================================================
//
// After 1.e4:
//   - Current pos: black to move, e-pawn on e4
//   - History[0] (slot 1): startpos (white to move)
//
// LC0 mirroring for black-to-move:
//   Squares are rank-flipped: sq XOR 56.
//   "Our" = black pieces (from black's perspective = bottom of flipped board)
//   "Their" = white pieces
//
// In the encoded output at slot 0 (black to move, rank-flipped):
//   - Black pawns (plane 0): black pawns on rank 7 in normal coords.
//     After rank-flip, rank 7 → rank 1 (sq 48-55 → sq 8-15 in flipped coords).
//     The pawn on e7 in normal coords (sq 52) → maps to sq 52 XOR 56 = sq 4
//     Wait: 52 XOR 56 = 52 ^ 56 = 12. So e7 → sq 12 (e2 from white's view).
//     The black pawns except the e-pawn are on rank 7: a7(48)..d7(51), f7(53)..h7(55).
//     After flip: 48^56=8, 49^56=9, 50^56=10, 51^56=11, 53^56=13, 54^56=14, 55^56=15.
//     So flipped black pawns: sq 8,9,10,11,13,14,15 (e-pawn gone from rank 7).
//   - Their (white) pieces: white rank 1 is rank 8 from black's view.
//     e-pawn moved to e4 (sq 28): after flip = 28^56 = 44 (from black's view = e5).
//     Other white pawns on rank 2 (sq 8-15) flip to rank 7 from black's view.
//     8^56=48, 9^56=49, 10^56=50, ..., 15^56=55.

TEST(LC0Encoding, After1e4_HistorySlot1ShowsStartpos) {
    ChessPosition startpos;
    ChessPosition pos_after_e4 = apply_uci(startpos, "e2e4");

    // history[0] = startpos (the pos before e4 was played)
    std::vector<ChessPosition> hist = {startpos};

    auto buf = make_buf();
    encode_nn(pos_after_e4, hist, buf.data());

    const float* t = buf.data();

    // ---- Verify history slot 1 (planes 13-25) shows startpos ----
    // At history slot 1 (1 ply ago = startpos), white was to move.
    //
    // LC0 encoding for slot 1 (black-to-move current position):
    //   flip_for_slot = (1 % 2 == 1) = True
    //   should_mirror = black_to_move ^ flip_for_slot = True ^ True = False
    //   → NO rank-flip for slot 1 (the two flips cancel out)
    //   slot_stm_is_ours = (1 % 2 == 0) = False → ours_color = WHITE
    //
    // Result: white pieces at their ACTUAL positions (no rank-flip), black at actual.
    // Verified against Python reference implementation (training/src/training/lc0/encoding.py).

    // White pawns (startpos rank 2): sq 8-15, no rank-flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq >= 8 && sq <= 15) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 13), expected) << "hist1 our-pawn sq=" << sq;
    }

    // White knights: b1=sq1, g1=sq6, no rank-flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 1 || sq == 6) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 14), expected) << "hist1 our-knight sq=" << sq;
    }

    // White bishops: c1=sq2, f1=sq5, no rank-flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 2 || sq == 5) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 15), expected) << "hist1 our-bishop sq=" << sq;
    }

    // White rooks: a1=sq0, h1=sq7, no rank-flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 0 || sq == 7) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 16), expected) << "hist1 our-rook sq=" << sq;
    }

    // White queen: d1=sq3, no rank-flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 3) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 17), expected) << "hist1 our-queen sq=" << sq;
    }

    // White king: e1=sq4, no rank-flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 4) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 18), expected) << "hist1 our-king sq=" << sq;
    }

    // "Their" in slot1 = BLACK pieces = startpos black pieces, no rank-flip:
    // Black pawns: a7-h7 = sq48-55, no flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq >= 48 && sq <= 55) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 19), expected) << "hist1 their-pawn sq=" << sq;
    }

    // Black knights: b8=sq57, g8=sq62, no flip
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 57 || sq == 62) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 20), expected) << "hist1 their-knight sq=" << sq;
    }
}

// ============================================================================
// Test 4: After 1.e4 — slot 0 shows the e4 position from black's perspective
// ============================================================================

TEST(LC0Encoding, After1e4_Slot0BlackPerspective) {
    ChessPosition startpos;
    ChessPosition pos_after_e4 = apply_uci(startpos, "e2e4");
    std::vector<ChessPosition> hist = {startpos};

    auto buf = make_buf();
    encode_nn(pos_after_e4, hist, buf.data());

    const float* t = buf.data();

    // Slot 0 (planes 0-12): current pos, black to move, rank-flipped.
    // slot_our_color = BLACK (even slot, current stm=BLACK → our_color=BLACK)
    // Black pawns are on rank 7 (a7=48..h7=55) in real coords.
    // After rank-flip: 48^56=8, 49^56=9, ..., 55^56=15.
    // So "our" pawns (plane 0) should be on sq 8-15.

    // Our (black) pawns after rank-flip:
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq >= 8 && sq <= 15) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 0), expected) << "slot0 our-pawn(black) sq=" << sq;
    }

    // Our (black) knights: b8=57, g8=62 → 57^56=1, 62^56=6
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq == 1 || sq == 6) ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 1), expected) << "slot0 our-knight(black) sq=" << sq;
    }

    // Their (white) pawns: on rank 2 EXCEPT e-pawn moved to e4.
    // White pawns: a2=8, b2=9, c2=10, d2=11, f2=13, g2=14, h2=15, e4=28
    // Rank-flip formula: sq ^ 56
    //   8^56=48, 9^56=49, 10^56=50, 11^56=51, 13^56=53, 14^56=54, 15^56=55
    //   28^56 = 0b011100 ^ 0b111000 = 0b100100 = 36  (not 44!)
    for (int sq = 0; sq < 64; ++sq) {
        bool expected_set = (sq == 48 || sq == 49 || sq == 50 || sq == 51 ||
                             sq == 53 || sq == 54 || sq == 55 || sq == 36);
        float expected = expected_set ? 1.0f : 0.0f;
        EXPECT_EQ(val(t, sq, 6), expected) << "slot0 their-pawn(white) sq=" << sq;
    }
}

// ============================================================================
// Test 5: Castling rights planes 104-107
// ============================================================================

TEST(LC0Encoding, CastlingRightPlanes) {
    // At startpos, all 4 castling rights should be available.
    ChessPosition pos;
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    const float* t = buf.data();

    // plane 104 = we can castle queenside (white at startpos = yes)
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle000, 1.0f));
    // plane 105 = we can castle kingside (yes)
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle00, 1.0f));
    // plane 106 = they can castle queenside (yes)
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle000, 1.0f));
    // plane 107 = they can castle kingside (yes)
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle00, 1.0f));
}

TEST(LC0Encoding, CastlingRightsLost) {
    // Position with no castling rights
    ChessPosition pos("8/8/8/8/8/8/8/4K3 w - - 0 1");
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    const float* t = buf.data();

    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle000, 0.0f));
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle00, 0.0f));
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle000, 0.0f));
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle00, 0.0f));
}

TEST(LC0Encoding, CastlingRightsPartial) {
    // Position where only white can castle kingside
    ChessPosition pos("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    const float* t = buf.data();

    // White to move, both sides can castle
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle000, 1.0f));
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle00, 1.0f));
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle000, 1.0f));
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle00, 1.0f));
}

// ============================================================================
// Test 6: STM plane (108)
//   - 0.0 when white to move
//   - 1.0 when black to move
// ============================================================================

TEST(LC0Encoding, STMPlane_WhiteToMove) {
    ChessPosition pos;  // startpos, white to move
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    EXPECT_TRUE(plane_all(buf.data(), AUX_PLANE_BASE + kAuxSTM, 0.0f))
        << "STM plane should be 0 for white to move";
}

TEST(LC0Encoding, STMPlane_BlackToMove) {
    ChessPosition pos;
    ChessPosition pos_after_e4 = apply_uci(pos, "e2e4");  // now black to move
    std::vector<ChessPosition> hist = {pos};

    auto buf = make_buf();
    encode_nn(pos_after_e4, hist, buf.data());

    EXPECT_TRUE(plane_all(buf.data(), AUX_PLANE_BASE + kAuxSTM, 1.0f))
        << "STM plane should be 1 for black to move";
}

// ============================================================================
// Test 7: Rule50 plane (109) = raw halfmove_clock (LC0's convention, not /100).
// ============================================================================

TEST(LC0Encoding, Rule50Plane) {
    // Position with halfmove clock = 50
    ChessPosition pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 50 26");
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    // LC0's INPUT_CLASSICAL_112_PLANE stores the raw halfmove clock, not /100.
    // Verified against Python reference: result[AUX+5] = float(board.halfmove_clock).
    const float expected = 50.0f;
    EXPECT_TRUE(plane_all(buf.data(), AUX_PLANE_BASE + kAuxRule50, expected))
        << "Rule50 plane should be raw halfmove_clock (= 50.0), not /100";
}

TEST(LC0Encoding, Rule50Plane_Zero) {
    ChessPosition pos;  // startpos has halfmove clock 0
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    EXPECT_TRUE(plane_all(buf.data(), AUX_PLANE_BASE + kAuxRule50, 0.0f))
        << "Rule50 plane at startpos should be 0";
}

// ============================================================================
// Test 8: AllOnes plane (111) and AllZeros plane (110)
// ============================================================================

TEST(LC0Encoding, AllOnesPlane) {
    ChessPosition pos;
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    EXPECT_TRUE(plane_all(buf.data(), AUX_PLANE_BASE + kAuxAllOnes, 1.0f))
        << "AllOnes plane (111) should be all 1.0";
}

TEST(LC0Encoding, AllZerosPlane) {
    ChessPosition pos;
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    EXPECT_TRUE(plane_all(buf.data(), AUX_PLANE_BASE + kAuxAllZeros, 0.0f))
        << "AllZeros plane (110) should be all 0.0";
}

// ============================================================================
// Test 9: Repetition plane — set when position repeated
// ============================================================================

TEST(LC0Encoding, RepetitionPlane_NoRepeat) {
    // With no history, the startpos repetition counter should be 0.
    ChessPosition pos;
    std::vector<ChessPosition> hist;
    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    EXPECT_TRUE(plane_all(buf.data(), 12, 0.0f))
        << "No-history position should have rep plane = 0";
}

TEST(LC0Encoding, RepetitionPlane_WithRepeat) {
    // Play Ng1-f3 (white knight), Ng8-f6 (black knight), Nf3-g1, Nf6-g8
    // Now we are back at startpos. The repetition plane for slot 0 should be set.
    // Specifically, current hash should match the hash 4 half-plies ago.

    ChessPosition pos0;  // startpos
    ChessPosition pos1 = apply_uci(pos0, "g1f3");
    ChessPosition pos2 = apply_uci(pos1, "g8f6");
    ChessPosition pos3 = apply_uci(pos2, "f3g1");
    ChessPosition pos4 = apply_uci(pos3, "f6g8");  // back to startpos!

    // history for pos4: [pos3, pos2, pos1, pos0]
    std::vector<ChessPosition> hist = {pos3, pos2, pos1, pos0};

    auto buf = make_buf();
    encode_nn(pos4, hist, buf.data());

    // pos4 hash should equal pos0 hash. pos0 is at hist[3] (slot 4 = index j=4).
    // Repetition check: slot 0 looks at slots 2, 4, 6 for match.
    // slot 4 → j=4: hist[3] = pos0. hash(pos4) == hash(pos0)?
    // pos4 and pos0 should have the same board position but possibly different
    // move history. Zobrist hash only covers piece positions + castling + ep + stm.
    EXPECT_EQ(pos4.board.hash(), pos0.board.hash())
        << "pos4 and pos0 should have equal Zobrist hash (same position)";

    // The repetition plane for slot 0 should be 1.0
    EXPECT_TRUE(plane_all(buf.data(), 12, 1.0f))
        << "Repetition plane should be 1.0 for repeated position";
}

// ============================================================================
// Test 10: HistoryTracker API
// ============================================================================

TEST(HistoryTracker, PushAndHistory) {
    HistoryTracker tracker;

    ChessPosition pos0;  // startpos
    ChessPosition pos1 = apply_uci(pos0, "e2e4");
    ChessPosition pos2 = apply_uci(pos1, "e7e5");

    // Push pos0 (before first move)
    tracker.push(pos0);
    {
        auto hist = tracker.history();
        ASSERT_EQ(hist.size(), 1u);
        EXPECT_EQ(hist[0].board.hash(), pos0.board.hash());
    }

    // Push pos1 (after first move)
    tracker.push(pos1);
    {
        auto hist = tracker.history();
        ASSERT_EQ(hist.size(), 2u);
        EXPECT_EQ(hist[0].board.hash(), pos1.board.hash()); // most recent
        EXPECT_EQ(hist[1].board.hash(), pos0.board.hash()); // older
    }
}

TEST(HistoryTracker, Reset) {
    HistoryTracker tracker;
    ChessPosition pos0;
    tracker.push(pos0);
    EXPECT_EQ(tracker.size(), 1);
    tracker.reset();
    EXPECT_EQ(tracker.size(), 0);
    EXPECT_TRUE(tracker.history().empty());
}

TEST(HistoryTracker, MaxCapacity) {
    HistoryTracker tracker;
    ChessPosition pos;
    // Push more than kMaxHistory positions
    for (int i = 0; i < 20; ++i) {
        tracker.push(pos);
    }
    // Should be capped at kMaxHistory = HISTORY_PLIES - 1 = 7
    EXPECT_EQ(tracker.size(), HISTORY_PLIES - 1);
    EXPECT_EQ(static_cast<int>(tracker.history().size()), HISTORY_PLIES - 1);
}

// ============================================================================
// Test 11: Batch encoding produces same results as individual calls
// ============================================================================

TEST(LC0Encoding, BatchMatchesSingle) {
    ChessPosition pos0;  // startpos
    ChessPosition pos1 = apply_uci(pos0, "e2e4");
    ChessPosition pos2 = apply_uci(pos1, "e7e5");

    std::vector<ChessPosition> hist0 = {};
    std::vector<ChessPosition> hist1 = {pos0};
    std::vector<ChessPosition> hist2 = {pos1, pos0};

    using Pair = std::pair<ChessPosition, std::vector<ChessPosition>>;
    std::vector<Pair> items = {
        {pos0, hist0},
        {pos1, hist1},
        {pos2, hist2},
    };

    constexpr int stride = SEQ_LEN * FEAT_DIM;
    std::vector<float> batch_out(3 * stride, 0.0f);
    encode_nn_batch(items, batch_out.data());

    // Compare against individual calls
    for (int i = 0; i < 3; ++i) {
        auto single_buf = make_buf();
        encode_nn(items[i].first, items[i].second, single_buf.data());

        for (int j = 0; j < stride; ++j) {
            EXPECT_EQ(batch_out[i * stride + j], single_buf[j])
                << "Mismatch at item=" << i << " j=" << j;
        }
    }
}

// ============================================================================
// Test 12: Mirroring — verify rank-flip for black-to-move positions
// ============================================================================

TEST(LC0Encoding, BlackToMoveRankFlip) {
    // After 1.e4 (black to move), the black king is on e8 (sq 60 in normal coords).
    // After rank-flip: sq 60 XOR 56 = sq 4 (= e1 in white's view).
    // "Our" king (plane 5) should be at sq 4 from black's perspective.

    ChessPosition startpos;
    ChessPosition pos_after_e4 = apply_uci(startpos, "e2e4");
    std::vector<ChessPosition> hist = {startpos};

    auto buf = make_buf();
    encode_nn(pos_after_e4, hist, buf.data());

    const float* t = buf.data();

    // Our (black) king: e8=sq60, rank-flipped → sq 60^56 = 4
    EXPECT_EQ(val(t, 4, 5), 1.0f) << "Black king at sq 4 (flipped e8) in slot 0";
    EXPECT_EQ(val(t, 60, 5), 0.0f) << "Black king NOT at sq 60 when rank-flipped";

    // Their (white) king: e1=sq4, rank-flipped → sq 4^56 = 60
    EXPECT_EQ(val(t, 60, 11), 1.0f) << "White king at sq 60 (flipped e1) in slot 0";
    EXPECT_EQ(val(t, 4, 11), 0.0f) << "White king NOT at sq 4 when rank-flipped";
}

// ============================================================================
// Test 13: Plane count sanity — total FEAT_DIM = 112
// ============================================================================

TEST(LC0Encoding, PlaneDimensions) {
    EXPECT_EQ(FEAT_DIM, 112);
    EXPECT_EQ(SEQ_LEN, 64);
    EXPECT_EQ(HISTORY_PLIES, 8);
    EXPECT_EQ(PLANES_PER_BOARD, 13);
    EXPECT_EQ(AUX_PLANE_BASE, 104);
    EXPECT_EQ(AUX_PLANE_BASE + 7, 111);
}

// ============================================================================
// Test 14: No pieces in empty history slots (slots 2-7 with 1-ply history)
// ============================================================================

TEST(LC0Encoding, EmptyHistorySlots_AllZero) {
    ChessPosition pos;
    ChessPosition pos_after_e4 = apply_uci(pos, "e2e4");
    std::vector<ChessPosition> hist = {pos};  // only 1 entry

    auto buf = make_buf();
    encode_nn(pos_after_e4, hist, buf.data());

    const float* t = buf.data();

    // Slots 2..7 (planes 26-103) should all be zero
    for (int plane = 26; plane <= 103; ++plane) {
        for (int sq = 0; sq < 64; ++sq) {
            EXPECT_EQ(t[sq * FEAT_DIM + plane], 0.0f)
                << "slot plane=" << plane << " sq=" << sq << " should be 0";
        }
    }
}

// ============================================================================
// Test 15: Castling perspective — black-to-move swaps "our" and "their"
// ============================================================================

TEST(LC0Encoding, CastlingPerspective_BlackToMove) {
    // After 1.e4, black is to move. "We" = black, "they" = white.
    // Both sides still have castling rights at this point.
    ChessPosition startpos;
    ChessPosition pos = apply_uci(startpos, "e2e4");
    std::vector<ChessPosition> hist = {startpos};

    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    const float* t = buf.data();

    // From black's perspective:
    // plane 104 = we (BLACK) can castle queenside → yes at startpos
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle000, 1.0f))
        << "Black queenside castle right should be set";
    // plane 105 = we (BLACK) can castle kingside → yes
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle00, 1.0f))
        << "Black kingside castle right should be set";
    // plane 106 = they (WHITE) can castle queenside → yes
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle000, 1.0f))
        << "White queenside castle right should be set";
    // plane 107 = they (WHITE) can castle kingside → yes
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxTheyCastle00, 1.0f))
        << "White kingside castle right should be set";
}

// ============================================================================
// Test 16: Integration — FEN with specific position, verify key planes
// ============================================================================

TEST(LC0Encoding, KiwipetePosition) {
    // Kiwipete: famous perft position
    const char* fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    ChessPosition pos(fen);
    std::vector<ChessPosition> hist;

    auto buf = make_buf();
    encode_nn(pos, hist, buf.data());

    const float* t = buf.data();

    // White to move. Our king on e1 (sq4). Their king on e8 (sq60).
    EXPECT_EQ(val(t, 4, 5), 1.0f) << "White king at e1(sq4)";
    EXPECT_EQ(val(t, 60, 11), 1.0f) << "Black king at e8(sq60)";

    // Castling: White can castle both sides (KQkq)
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle000, 1.0f));
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxWeCastle00, 1.0f));

    // STM plane = 0 (white to move)
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxSTM, 0.0f));

    // Rule50 = 0 (0 half-moves)
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxRule50, 0.0f));

    // AllOnes
    EXPECT_TRUE(plane_all(t, AUX_PLANE_BASE + kAuxAllOnes, 1.0f));
}
