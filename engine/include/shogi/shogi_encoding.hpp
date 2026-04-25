// SPDX-License-Identifier: GPL-3.0-or-later
// engine/include/shogi/shogi_encoding.hpp
//
// Shogi NN input encoding — implements encoding_spec.md §3.
//
// LICENSE NOTE: This file includes cshogi headers (GPL v3).
// Only link into shogi_engine, never chess_engine.
//
// Tensor shape: [81, 90] float32
// Square index convention: (9 - file) * 9 + (rank - 1)
//   where file ∈ {1..9}, rank ∈ {1..9} (先手 perspective, file 9 leftmost).
// This matches cshogi's SQ99 - sq transformation.
//
// v1 active planes:
//   0..13  own-side pieces       (14 planes)
//   14..27 opponent-side pieces  (14 planes)
//   56..62 own hand pieces       (7 planes, broadcast)
//   63..69 opponent hand pieces  (7 planes, broadcast)
//   89     side-to-move          (1 plane,  broadcast)
// All other channels are zero-filled.
//
// §3.3 mirroring: when 後手 (White) is to move, rotate board 180° and swap
// colors before encoding so that "own side" is always the side to move.

#pragma once

#include "shogi/shogi_traits.hpp"

// cshogi headers already included via shogi_traits.hpp

#include <cstring>
#include <cassert>

namespace shogi {
namespace encoding {

// ---------------------------------------------------------------------------
// Square-index conversion
//
// cshogi square: makeSquare(file, rank) = file * 9 + rank
//   where File1=0..File9=8, Rank1=0..Rank9=8
//
// Our spec (encoding_spec.md §3.1):
//   square_index = (9 - file) * 9 + (rank - 1)
//   = (8 - cshogi_file) * 9 + cshogi_rank
//   = 80 - cshogi_file * 9 + cshogi_rank - 0   ... or more simply:
//
// Verify: cshogi SQ11 = File1=0, Rank1=0 → spec idx = 8*9+0 = 72.  ✓
//         cshogi SQ91 = File9=8, Rank1=0 → spec idx = 0*9+0 = 0.   ✓
//         cshogi SQ19 = File1=0, Rank9=8 → spec idx = 8*9+8 = 80.  ✓
//         cshogi SQ99 = File9=8, Rank9=8 → spec idx = 0*9+8 = 8.   ✓
//
// Relationship:
//   spec_sq = (8 - file_0) * 9 + rank_0
//           = SQ99 - (file_0 * 9) - (8 - rank_0)   [not simply SQ99-sq]
//
// Actually SQ99 - cshogi_sq = (8*9+8) - (file*9 + rank)
//                           = (8-file)*9 + (8-rank)   = mirror(file)*9 + mirror(rank)
// That is the 180-degree rotation (both file and rank mirrored), not what we want.
//
// We want (8-file)*9 + rank = file_mirrored * 9 + rank_same.
// This is NOT the same as SQ99 - sq.
//
// Correct formula:  spec_sq = (8 - makeFile(cshogi_sq)) * 9 + makeRank(cshogi_sq)
// ---------------------------------------------------------------------------
inline int cshogi_to_spec_sq(Square csq) {
    // File and rank are 0-indexed in cshogi (File1=0, Rank1=0).
    const int f = static_cast<int>(makeFile(csq));  // 0..8
    const int r = static_cast<int>(makeRank(csq));  // 0..8
    return (8 - f) * 9 + r;
}

// Inverse: spec square index → cshogi Square.
inline Square spec_to_cshogi_sq(int spec_sq) {
    // spec_sq = (8 - f) * 9 + r  →  f = 8 - spec_sq/9,  r = spec_sq % 9
    const int f = 8 - spec_sq / 9;
    const int r = spec_sq % 9;
    return makeSquare(static_cast<File>(f), static_cast<Rank>(r));
}

// When mirroring (後手 perspective): spec square s maps to 80 - s
// (both file and rank are inverted in the spec-sq space, which corresponds
// to a 180° board rotation + file swap for the opponent's view).
// Wait — let's think carefully.
//
// After mirroring (White to move): we swap colors and rotate board 180°.
// The new spec square for a piece at cshogi_sq (from White's POV):
//   mirrored cshogi sq = SQ99 - cshogi_sq
//   spec_mirrored = cshogi_to_spec_sq(SQ99 - cshogi_sq)
//
// Simplification:
//   cshogi_to_spec_sq(SQ99 - sq) = (8 - makeFile(SQ99-sq)) * 9 + makeRank(SQ99-sq)
//   = (8 - (8-f)) * 9 + (8-r) = f*9 + (8-r)
//
// And the original spec_sq for sq = (8-f)*9 + r.
// Mirrored spec_sq = f*9 + (8-r).
// Note: f*9 + (8-r) + (8-f)*9 + r = 8*9 + 8 = 80.
// So: mirrored_spec_sq = 80 - spec_sq.  ✓
//
// This confirms: mirrored spec square = 80 - original spec square.
inline int mirror_spec_sq(int spec_sq) {
    return 80 - spec_sq;
}

// ---------------------------------------------------------------------------
// Piece-type plane indices (encoding_spec.md §3.2)
// dlshogi 14-plane order:
//   0: Pawn, 1: Lance, 2: Knight, 3: Silver, 4: Bishop, 5: Rook,
//   6: Gold, 7: King, 8: ProPawn (と), 9: ProLance (成香), 10: ProKnight (成桂),
//   11: ProSilver (成銀), 12: Horse (馬), 13: Dragon (龍)
// ---------------------------------------------------------------------------
inline int piece_type_to_plane(PieceType pt) {
    // PieceType order in cshogi: Pawn=1,Lance=2,Knight=3,Silver=4,Bishop=5,
    // Rook=6,Gold=7,King=8,ProPawn=9,ProLance=10,ProKnight=11,ProSilver=12,
    // Horse=13,Dragon=14.
    // Plane index = pt - 1  (since Pawn=1 maps to plane 0).
    return static_cast<int>(pt) - 1;  // valid for pt in [Pawn, Dragon] = [1,14]
}

// HandPiece order: HPawn=0, HLance=1, HKnight=2, HSilver=3, HGold=4,
//                 HBishop=5, HRook=6.
// Matches encoding_spec.md §3.2 hand-piece order (歩香桂銀金角飛).
// Hand planes start at 56 (own) and 63 (opponent) in the 90-channel layout.
static constexpr int kOwnHandBase  = 56;
static constexpr int kOppHandBase  = 63;
static constexpr int kSideToMoveChannel = 89;

// Maximum number of each hand piece (for normalisation).
// Per cshogi.h MAX_PIECES_IN_HAND.
static constexpr float kMaxHand[HandPieceNum] = {
    8.0f,  // HPawn
    4.0f,  // HLance
    4.0f,  // HKnight
    4.0f,  // HSilver
    4.0f,  // HGold
    2.0f,  // HBishop
    2.0f,  // HRook
};

// ---------------------------------------------------------------------------
// encode_nn — main encoding function.
//
// out: pointer to float array of size 81 * 90 = 7290, initialised to 0.
// Layout: [sq_idx * 90 + channel] — row = square (slow), col = channel (fast).
// ---------------------------------------------------------------------------
inline void encode_nn(const ShogiPosition& wrapped_pos, float* out) {
    const ::Position& pos = wrapped_pos.raw();

    // Zero-fill the entire tensor.
    std::memset(out, 0, sizeof(float) * 81 * 90);

    const Color stm = pos.turn();   // side to move

    // § 3.3 mirroring: determine own/opp colors and square transform.
    // When Black to move: own=Black, opp=White, sq unchanged.
    // When White to move: own=White, opp=Black, sq mirrored.
    const bool mirror   = (stm == White);
    const Color own_col = stm;                            // side to move
    const Color opp_col = oppositeColor(stm);             // opponent

    // ---- Piece placement planes (channels 0..27) ----
    // Channels 0..13: own-side pieces.
    // Channels 14..27: opponent-side pieces.
    for (Square csq = SQ11; csq < SquareNum; ++csq) {
        const Piece p = pos.piece(csq);
        if (p == Empty) continue;

        const Color pc = pieceToColor(p);
        const PieceType pt = pieceToPieceType(p);

        // Determine our spec square index (with optional mirroring).
        int spec_sq = cshogi_to_spec_sq(csq);
        if (mirror) spec_sq = mirror_spec_sq(spec_sq);

        int plane;
        if (pc == own_col) {
            plane = piece_type_to_plane(pt);        // 0..13
        } else {
            plane = 14 + piece_type_to_plane(pt);   // 14..27
        }

        out[spec_sq * 90 + plane] = 1.0f;
    }

    // ---- Hand piece planes (channels 56..69, broadcast over all 81 squares) ----
    // Own hand: channels 56..62.
    // Opp hand: channels 63..69.
    for (int hp_i = 0; hp_i < HandPieceNum; ++hp_i) {
        const HandPiece hp = static_cast<HandPiece>(hp_i);

        const float own_cnt = static_cast<float>(
            pos.hand(own_col).numOf(hp));
        const float opp_cnt = static_cast<float>(
            pos.hand(opp_col).numOf(hp));

        // Normalise by max count (encoding_spec.md §3.2 says "count,
        // normalized by max").
        const float own_val = own_cnt / kMaxHand[hp_i];
        const float opp_val = opp_cnt / kMaxHand[hp_i];

        if (own_val > 0.0f) {
            const int ch_own = kOwnHandBase + hp_i;
            for (int sq = 0; sq < 81; ++sq)
                out[sq * 90 + ch_own] = own_val;
        }
        if (opp_val > 0.0f) {
            const int ch_opp = kOppHandBase + hp_i;
            for (int sq = 0; sq < 81; ++sq)
                out[sq * 90 + ch_opp] = opp_val;
        }
    }

    // ---- Side-to-move plane (channel 89, broadcast) ----
    // 1.0 = 先手 (Black) to move (in the mirrored frame, own side is always
    // conceptually "Black", so broadcast 1.0 when original stm == Black).
    // When White to move with mirroring, the board is flipped so the model
    // still sees "own side", and channel 89 = 0 indicates post-mirror White.
    const float stm_val = (stm == Black) ? 1.0f : 0.0f;
    for (int sq = 0; sq < 81; ++sq)
        out[sq * 90 + kSideToMoveChannel] = stm_val;
}

} // namespace encoding
} // namespace shogi
