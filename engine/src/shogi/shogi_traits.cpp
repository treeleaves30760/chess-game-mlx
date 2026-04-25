// SPDX-License-Identifier: GPL-3.0-or-later
// engine/src/shogi/shogi_traits.cpp
//
// Out-of-line definitions for ShogiTraits, ShogiPosition, ShogiMove,
// ShogiRuntime, and the 2187-slot policy mapping.
//
// LICENSE NOTE: This translation unit links against cshogi (GPL v3) and is
// therefore also GPL v3.  Only link into shogi_engine, never chess_engine.
//
// Policy mapping (encoding_spec.md §3.4):
//   2187 = 81 squares × 27 move types
//   Move types:
//     0..9   : 10 directions (non-promote)
//     10..19 : 10 directions (promote)
//     20..26 : 7 drop types (歩香桂銀金角飛)
//
//   Direction order (matches cshogi.h MOVE_DIRECTION enum):
//     0: UP, 1: UP_LEFT, 2: UP_RIGHT, 3: LEFT, 4: RIGHT,
//     5: DOWN, 6: DOWN_LEFT, 7: DOWN_RIGHT, 8: UP2_LEFT, 9: UP2_RIGHT
//     (10..19 = promoted versions of 0..9)
//
//   Drop type index: HPawn=0..HRook=6 (same as HandPiece enum order).

#include "shogi/shogi_traits.hpp"
#include "shogi/shogi_encoding.hpp"

#include <mutex>
#include <stdexcept>
#include <cassert>

// cshogi global-init function
extern void initTable();

namespace shogi {

// ---------------------------------------------------------------------------
// ShogiRuntime
// ---------------------------------------------------------------------------

std::once_flag ShogiRuntime::s_init_flag;

void ShogiRuntime::ensure_initialized() {
    std::call_once(s_init_flag, []() {
        initTable();           // initialises Bitboards, magic tables, Zobrist
        Position::initZobrist();
        initMate1Ply();
        HuffmanCodedPos::init();
    });
}

// ---------------------------------------------------------------------------
// ShogiPosition
// ---------------------------------------------------------------------------

ShogiPosition::ShogiPosition(const std::string& sfen) {
    ShogiRuntime::ensure_initialized();
    pos_.set(sfen);
    // Push a sentinel StateInfo for undoMove compatibility.
    // (cshogi's StateInfo chain starts from pos_.st_ which points to
    // pos_.startState_ — we do not need an extra sentinel here.)
}

ShogiPosition::ShogiPosition(const ShogiPosition& other)
    : pos_(other.pos_), state_stack_(other.state_stack_)
{
    // After copying, pos_.st_ may point into the original's startState_.
    // Re-point it into our own startState_ by replaying the state chain.
    // cshogi's operator= handles this correctly.
}

ShogiPosition& ShogiPosition::operator=(const ShogiPosition& other) {
    if (this != &other) {
        pos_         = other.pos_;
        state_stack_ = other.state_stack_;
    }
    return *this;
}

void ShogiPosition::do_move(ShogiMove m) {
    state_stack_.emplace_back();
    pos_.doMove(m.raw(), state_stack_.back());
}

void ShogiPosition::undo_move(ShogiMove m) {
    assert(!state_stack_.empty());
    pos_.undoMove(m.raw());
    state_stack_.pop_back();
}

// ---------------------------------------------------------------------------
// Policy mapping helpers
// ---------------------------------------------------------------------------

// Direction table matching MOVE_DIRECTION enum in cshogi.h.
// Index → (delta_file, delta_rank) from Black's perspective.
// File increases rightward (East), Rank increases downward (South) in cshogi.
// "UP" for Black = decreasing rank = DeltaN = (-1) in rank.
//
// cshogi Square layout: sq = file*9 + rank.
// DeltaN = -1 (rank decreases = piece moves up toward rank 1).
// DeltaE = -9 (file decreases = piece moves right toward file 1).
// DeltaS = +1, DeltaW = +9.
//
// For the policy encoding, we describe the move direction as seen by the
// moving piece (already in the "own side" frame after potential mirroring).
// We use direction in terms of (to_file - from_file, to_rank - from_rank)
// in cshogi coordinates.

struct DirEntry {
    int df;  // to_file - from_file (in cshogi 0-indexed coords)
    int dr;  // to_rank - from_rank
};

// Matches MOVE_DIRECTION order: UP=0..UP2_RIGHT=9.
// "UP" from Black's perspective = rank decreases (negative dr).
static const DirEntry kDirections[10] = {
    {  0, -1 },  // 0: UP
    { -1, -1 },  // 1: UP_LEFT
    {  1, -1 },  // 2: UP_RIGHT
    { -1,  0 },  // 3: LEFT
    {  1,  0 },  // 4: RIGHT
    {  0,  1 },  // 5: DOWN
    { -1,  1 },  // 6: DOWN_LEFT
    {  1,  1 },  // 7: DOWN_RIGHT
    { -1, -2 },  // 8: UP2_LEFT  (knight move: left+2up)
    {  1, -2 },  // 9: UP2_RIGHT (knight move: right+2up)
};

// Given two squares (in cshogi coordinates), compute the normalised direction
// index (0..9) or return -1 if not on the direction list.
// For sliding pieces, the direction is normalised to unit step.
static int get_direction_idx(Square from_sq, Square to_sq) {
    const int from_f = static_cast<int>(makeFile(from_sq));
    const int from_r = static_cast<int>(makeRank(from_sq));
    const int to_f   = static_cast<int>(makeFile(to_sq));
    const int to_r   = static_cast<int>(makeRank(to_sq));

    int df = to_f - from_f;
    int dr = to_r - from_r;

    // Normalise sliding direction to unit step.
    if (df == 0 && dr == 0) return -1;

    if (std::abs(df) == std::abs(dr)) {
        // Diagonal slider.
        df = (df > 0) ? 1 : -1;
        dr = (dr > 0) ? 1 : -1;
    } else if (df == 0) {
        dr = (dr > 0) ? 1 : -1;
    } else if (dr == 0) {
        df = (df > 0) ? 1 : -1;
    }
    // Knight moves: df=±1, dr=-2 — already unit, no normalisation needed.

    for (int i = 0; i < 10; ++i) {
        if (kDirections[i].df == df && kDirections[i].dr == dr)
            return i;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// move_to_policy_idx
//
// Implements encoding_spec.md §3.4:
//   policy_idx = to_spec_sq * 27 + move_type_idx
//
// Move type encoding:
//   0..9   : board moves by direction (no promotion)
//   10..19 : board moves by direction (with promotion)
//   20..26 : drops by piece type (HPawn=20 .. HRook=26)
//
// The "to" square is in the mirrored spec-sq space if White is to move.
// ---------------------------------------------------------------------------

int ShogiTraits::move_to_policy_idx(ShogiMove sm, const ShogiPosition& wrapped_pos) {
    const ::Move m = sm.raw();
    const ::Position& pos = wrapped_pos.raw();
    const bool mirror = (pos.turn() == White);

    if (m.isDrop()) {
        // Drop move: move type = 20 + HandPiece index.
        // from() for drops = SquareNum - 1 + PieceType, so:
        const PieceType pt = m.pieceTypeDropped();
        const HandPiece hp = pieceTypeToHandPiece(pt);
        const int mt = 20 + static_cast<int>(hp);

        int to_spec = encoding::cshogi_to_spec_sq(m.to());
        if (mirror) to_spec = encoding::mirror_spec_sq(to_spec);

        return to_spec * 27 + mt;
    }

    // Board move.
    Square from_sq = m.from();
    Square to_sq   = m.to();

    // For White (mirrored), flip squares before computing direction.
    if (mirror) {
        from_sq = static_cast<Square>(SQ99 - from_sq);
        to_sq   = static_cast<Square>(SQ99 - to_sq);
    }

    const int dir_idx = get_direction_idx(from_sq, to_sq);
    assert(dir_idx >= 0 && "move_to_policy_idx: unrecognised direction");

    const int promote_offset = m.isPromotion() ? 10 : 0;
    const int mt = dir_idx + promote_offset;

    // to_spec is in the (possibly mirrored) frame.
    int to_spec;
    if (mirror) {
        to_spec = encoding::cshogi_to_spec_sq(static_cast<Square>(SQ99 - m.to()));
    } else {
        to_spec = encoding::cshogi_to_spec_sq(m.to());
    }

    return to_spec * 27 + mt;
}

// ---------------------------------------------------------------------------
// policy_idx_to_move
// ---------------------------------------------------------------------------

ShogiMove ShogiTraits::policy_idx_to_move(int idx, const ShogiPosition& wrapped_pos) {
    assert(idx >= 0 && idx < 2187);

    const ::Position& pos = wrapped_pos.raw();
    const bool mirror = (pos.turn() == White);

    const int to_spec = idx / 27;
    const int mt      = idx % 27;

    if (mt >= 20) {
        // Drop move.
        const int hp_idx = mt - 20;
        const HandPiece hp = static_cast<HandPiece>(hp_idx);
        const PieceType pt = handPieceToPieceType(hp);

        int spec = to_spec;
        if (mirror) spec = encoding::mirror_spec_sq(spec);
        const Square to_csq = encoding::spec_to_cshogi_sq(spec);

        return ShogiMove(makeDropMove(pt, to_csq));
    }

    // Board move.
    const bool promote = (mt >= 10);
    const int dir_idx = promote ? (mt - 10) : mt;

    const DirEntry& d = kDirections[dir_idx];

    // Reconstruct to_sq in possibly-mirrored cshogi frame.
    int spec = to_spec;
    if (mirror) spec = encoding::mirror_spec_sq(spec);
    const Square to_csq = encoding::spec_to_cshogi_sq(spec);

    const int to_f = static_cast<int>(makeFile(to_csq));
    const int to_r = static_cast<int>(makeRank(to_csq));

    // Derive the scan direction to find from_sq in the real (unmirrored) position.
    //
    // Encoding invariant (derived from the mirror transform):
    //   Let d = kDirections[dir_idx] (direction in the encoded frame).
    //   For Black (no mirror): real_from → real_to in direction d.
    //     real_from = real_to - d  → scan backward with step (-d).
    //   For White (mirror): real_to - real_from = d  →  real_from = real_to + d
    //     → scan backward with step (+d).
    //
    // In both cases we sweep from real_to until we find the non-empty from_sq.
    const int step_df = mirror ? d.df : -d.df;
    const int step_dr = mirror ? d.dr : -d.dr;

    // For knight moves (dir_idx 8,9), exactly one step.
    if (dir_idx == 8 || dir_idx == 9) {
        const int from_f = to_f + step_df;
        const int from_r = to_r + step_dr;
        if (!isInSquare(static_cast<File>(from_f), static_cast<Rank>(from_r)))
            return ShogiMove();
        const Square from_csq = makeSquare(static_cast<File>(from_f),
                                            static_cast<Rank>(from_r));
        ::Move raw = makeMove(pieceToPieceType(pos.piece(from_csq)), from_csq, to_csq);
        if (promote) raw |= promoteFlag();
        return ShogiMove(raw);
    }

    // Slider / step piece: scan along step direction until we find a non-empty square.
    {
        int cur_f = to_f + step_df;
        int cur_r = to_r + step_dr;
        while (isInSquare(static_cast<File>(cur_f), static_cast<Rank>(cur_r))) {
            const Square cur_sq = makeSquare(static_cast<File>(cur_f),
                                              static_cast<Rank>(cur_r));
            const ::Piece p = pos.piece(cur_sq);
            if (p != Empty) {
                ::Move raw = makeMove(pieceToPieceType(p), cur_sq, to_csq);
                if (promote) raw |= promoteFlag();
                return ShogiMove(raw);
            }
            cur_f += step_df;
            cur_r += step_dr;
        }
    }

    // No piece found — return an invalid move.
    return ShogiMove();
}

// ---------------------------------------------------------------------------
// ShogiTraits — GameRules interface implementations
// ---------------------------------------------------------------------------

void ShogiTraits::generate_legal(const ShogiPosition& wrapped_pos,
                                  core::MoveList<ShogiMove>& out) {
    ShogiRuntime::ensure_initialized();
    out.clear();
    const MoveList<LegalAll> ml(wrapped_pos.raw());
    out.moves.reserve(ml.size());
    for (const ExtMove* it = const_cast<MoveList<LegalAll>&>(ml).begin();
         it != const_cast<MoveList<LegalAll>&>(ml).begin() + ml.size(); ++it) {
        out.push(ShogiMove(it->move));
    }
}

void ShogiTraits::apply(ShogiPosition& pos, ShogiMove m) {
    pos.do_move(m);
}

void ShogiTraits::undo(ShogiPosition& pos, ShogiMove m) {
    pos.undo_move(m);
}

bool ShogiTraits::is_terminal(const ShogiPosition& pos) {
    // Terminal if the current player has no legal moves (= loss).
    // Also treat repetition draw as terminal.
    const MoveList<LegalAll> ml(pos.raw());
    if (ml.size() == 0) return true;
    const RepetitionType rep = pos.raw().isDraw();
    return rep != NotRepetition;
}

float ShogiTraits::terminal_value(const ShogiPosition& pos) {
    const RepetitionType rep = pos.raw().isDraw();
    if (rep == RepetitionDraw) return 0.0f;
    // No legal moves: current player loses.
    return -1.0f;
}

std::uint64_t ShogiTraits::hash(const ShogiPosition& pos) {
    return pos.key();
}

void ShogiTraits::encode_nn(const ShogiPosition& pos, float* out) {
    shogi::encoding::encode_nn(pos, out);
}

} // namespace shogi
