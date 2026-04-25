#pragma once

// ----------------------------------------------------------------------------
// engine/include/chess/chess_encoding.hpp
//
// Implements encode_nn() and move_to_policy_idx() for chess positions.
//
// Input tensor: [64, 19] float32, row-major.
//   Square index: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.
//   Feature planes per square: see encoding_spec.md §1.
//
// Policy index scheme:
//   We use the full 64×73 = 4672-slot space here (simple, no compact table
//   needed at the MCTS layer).  The compact 1858-slot LC0 mapping is applied
//   at the NN boundary (training/inference); the C++ engine always works with
//   4672-slot indices.
//
//   Deferral note: compact_1858.hpp will map 4672 → 1858 in Phase 3 when the
//   actual NN is loaded.  For Phase 1, correctness of move_to_policy_idx is
//   verified by the round-trip property: policy_idx_to_move(move_to_policy_idx(m)) == m.
//
// Move-type encoding (73 slots per from-square):
//   Slots 0..55  — queen-like rays: direction d (0..7) × distance k (1..7)
//                  index = d*7 + (k-1)
//   Slots 56..63 — knight jumps (8 fixed offsets, ordered by delta)
//   Slots 64..72 — underpromotions: direction (0 left-diagonal, 1 forward,
//                  2 right-diagonal) × piece (0 knight, 1 bishop, 2 rook)
//                  index = 64 + dir*3 + piece_offset
//   Queen-promotion uses the corresponding queen-ray slot (no separate slot).
//
// See also: shared/encoding_spec.md §2
// ----------------------------------------------------------------------------

#include <chess.hpp>
#include <array>
#include <cstring>

// Forward declaration — defined in chess_traits.hpp
namespace chess_mlx::chess {
struct ChessPosition;
struct ChessMove;
} // namespace chess_mlx::chess

namespace chess_mlx::chess {

// ============================================================================
// Constants
// ============================================================================

static constexpr int kSquares         = 64;
static constexpr int kInputChannels   = 19;
static constexpr int kInputSize       = kSquares * kInputChannels; // 1216

// Policy layout
static constexpr int kMoveDirsQueen   = 8;
static constexpr int kMoveDistances   = 7;
static constexpr int kQueenSlots      = kMoveDirsQueen * kMoveDistances; // 56
static constexpr int kKnightSlots     = 8;
static constexpr int kUnderpromoSlots = 9; // 3 dirs × 3 pieces (not queen)
static constexpr int kMovetypes       = kQueenSlots + kKnightSlots + kUnderpromoSlots; // 73
static constexpr int kPolicySize      = kSquares * kMovetypes; // 4672

// Feature plane indices (matching encoding_spec.md §1.2)
static constexpr int kFeatWhitePawn   = 0;
static constexpr int kFeatWhiteKnight = 1;
static constexpr int kFeatWhiteBishop = 2;
static constexpr int kFeatWhiteRook   = 3;
static constexpr int kFeatWhiteQueen  = 4;
static constexpr int kFeatWhiteKing   = 5;
static constexpr int kFeatBlackPawn   = 6;
static constexpr int kFeatBlackKnight = 7;
static constexpr int kFeatBlackBishop = 8;
static constexpr int kFeatBlackRook   = 9;
static constexpr int kFeatBlackQueen  = 10;
static constexpr int kFeatBlackKing   = 11;
static constexpr int kFeatSideToMove  = 12;
static constexpr int kFeatWKCastle    = 13;
static constexpr int kFeatWQCastle    = 14;
static constexpr int kFeatBKCastle    = 15;
static constexpr int kFeatBQCastle    = 16;
static constexpr int kFeatEnPassant   = 17;
static constexpr int kFeatHalfMove    = 18;

// ============================================================================
// Direction tables for queen-like rays
// ============================================================================
//
// 8 directions:  N, NE, E, SE, S, SW, W, NW
// delta-file, delta-rank per step:
//   dir 0: N  (0, +1)
//   dir 1: NE (+1, +1)
//   dir 2: E  (+1,  0)
//   dir 3: SE (+1, -1)
//   dir 4: S  (0, -1)
//   dir 5: SW (-1, -1)
//   dir 6: W  (-1,  0)
//   dir 7: NW (-1, +1)
static constexpr int kDirDeltaFile[8] = {  0, +1, +1, +1,  0, -1, -1, -1 };
static constexpr int kDirDeltaRank[8] = { +1, +1,  0, -1, -1, -1,  0, +1 };

// Knight jump deltas (file, rank) — 8 fixed offsets in a canonical order
static constexpr int kKnightDeltaFile[8] = { -2, -1, +1, +2, +2, +1, -1, -2 };
static constexpr int kKnightDeltaRank[8] = { +1, +2, +2, +1, -1, -2, -2, -1 };

// ============================================================================
// Internal helpers
// ============================================================================

namespace detail {

/// Return feature-plane index [0..11] for a chess::Piece, or -1 for NONE.
inline int piece_to_feature_plane(const ::chess::Piece p) noexcept {
    if (p == ::chess::Piece::NONE) return -1;
    const int color_offset = (p.color() == ::chess::Color::BLACK) ? 6 : 0;
    // PieceType: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
    return color_offset + static_cast<int>(p.type().internal());
}

/// Encode the queen-like move slot for (from_sq, to_sq).
/// Returns -1 if it is not a queen-like move.
inline int queen_ray_slot(int from_sq, int to_sq) noexcept {
    const int from_file = from_sq & 7;
    const int from_rank = from_sq >> 3;
    const int to_file   = to_sq & 7;
    const int to_rank   = to_sq >> 3;
    const int df = to_file - from_file;
    const int dr = to_rank - from_rank;

    // Must be along a ray (same file, rank, or diagonal)
    if (df == 0 && dr == 0) return -1;
    if (df != 0 && dr != 0 && std::abs(df) != std::abs(dr)) return -1;

    // Determine direction
    const int sdf = (df > 0) - (df < 0);
    const int sdr = (dr > 0) - (dr < 0);

    int dir = -1;
    for (int d = 0; d < 8; ++d) {
        if (kDirDeltaFile[d] == sdf && kDirDeltaRank[d] == sdr) {
            dir = d;
            break;
        }
    }
    if (dir < 0) return -1;

    const int dist = std::max(std::abs(df), std::abs(dr)); // 1..7
    return dir * kMoveDistances + (dist - 1);
}

/// Encode the knight-jump slot for (from_sq, to_sq).
/// Returns -1 if not a knight move.
inline int knight_slot(int from_sq, int to_sq) noexcept {
    const int from_file = from_sq & 7;
    const int from_rank = from_sq >> 3;
    const int df = (to_sq & 7) - from_file;
    const int dr = (to_sq >> 3) - from_rank;

    for (int k = 0; k < 8; ++k) {
        if (kKnightDeltaFile[k] == df && kKnightDeltaRank[k] == dr) {
            return kQueenSlots + k;
        }
    }
    return -1;
}

/// Encode the underpromotion slot for a pawn promotion to piece `pt` (not QUEEN).
/// Returns -1 if this is a queen promotion (use the queen-ray slot instead).
/// `from_sq` and `to_sq` are the pawn's squares.
inline int underpromo_slot(int from_sq, int to_sq, ::chess::PieceType pt) noexcept {
    if (pt == ::chess::PieceType::QUEEN) return -1;

    const int df = (to_sq & 7) - (from_sq & 7);
    // df: -1 = left-diagonal, 0 = forward, +1 = right-diagonal
    int dir = -1;
    if (df == -1) dir = 0;
    else if (df == 0) dir = 1;
    else if (df == +1) dir = 2;
    else return -1;

    int piece_offset = -1;
    switch (pt.internal()) {
        case ::chess::PieceType::KNIGHT: piece_offset = 0; break;
        case ::chess::PieceType::BISHOP: piece_offset = 1; break;
        case ::chess::PieceType::ROOK:   piece_offset = 2; break;
        default: return -1;
    }
    return kQueenSlots + kKnightSlots + dir * 3 + piece_offset;
}

} // namespace detail

// ============================================================================
// Public API
// ============================================================================

/// Encode position `board` into `out` ([64 × 19] float32, row-major).
/// `out` must point to at least kInputSize = 1216 floats.
/// All values are pure 0.0f or 1.0f except feature 18 (normalised half-move clock).
inline void encode_nn(const ::chess::Board& board, float* out) noexcept {
    std::memset(out, 0, static_cast<std::size_t>(kInputSize) * sizeof(float));

    // --- Piece planes (features 0..11) ---
    for (int sq = 0; sq < kSquares; ++sq) {
        const auto piece = board.at(::chess::Square(sq));
        const int feat   = detail::piece_to_feature_plane(piece);
        if (feat >= 0) {
            out[sq * kInputChannels + feat] = 1.0f;
        }
    }

    // --- Scalar features broadcast to all 64 squares ---
    const bool white_to_move = (board.sideToMove() == ::chess::Color::WHITE);
    const auto cr            = board.castlingRights();

    const float stm_val  = white_to_move ? 1.0f : 0.0f;
    const float wk_val   = cr.has(::chess::Color::WHITE, ::chess::Board::CastlingRights::Side::KING_SIDE)  ? 1.0f : 0.0f;
    const float wq_val   = cr.has(::chess::Color::WHITE, ::chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    const float bk_val   = cr.has(::chess::Color::BLACK, ::chess::Board::CastlingRights::Side::KING_SIDE)  ? 1.0f : 0.0f;
    const float bq_val   = cr.has(::chess::Color::BLACK, ::chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    const float hm_val   = static_cast<float>(board.halfMoveClock()) / 100.0f;

    for (int sq = 0; sq < kSquares; ++sq) {
        float* base = out + sq * kInputChannels;
        base[kFeatSideToMove] = stm_val;
        base[kFeatWKCastle]   = wk_val;
        base[kFeatWQCastle]   = wq_val;
        base[kFeatBKCastle]   = bk_val;
        base[kFeatBQCastle]   = bq_val;
        base[kFeatHalfMove]   = hm_val;
    }

    // --- En passant target square (feature 17) ---
    const auto ep_sq = board.enpassantSq();
    if (ep_sq != ::chess::Square::NO_SQ) {
        const int ep_idx = ep_sq.index();
        out[ep_idx * kInputChannels + kFeatEnPassant] = 1.0f;
    }
}

/// Map a chess::Move to the 4672-slot policy index.
/// Returns -1 on failure (should never happen for legal moves).
inline int move_to_policy_idx(const ::chess::Move& mv) noexcept {
    const int from = mv.from().index();
    const int to   = mv.to().index();

    if (mv.typeOf() == ::chess::Move::CASTLING) {
        // Castling is encoded as a queen-ray move (king slides along rank).
        const int slot = detail::queen_ray_slot(from, to);
        if (slot < 0) return -1;
        return from * kMovetypes + slot;
    }

    if (mv.typeOf() == ::chess::Move::ENPASSANT) {
        // Encoded as a diagonal queen-ray move.
        const int slot = detail::queen_ray_slot(from, to);
        if (slot < 0) return -1;
        return from * kMovetypes + slot;
    }

    if (mv.typeOf() == ::chess::Move::PROMOTION) {
        const auto pt = mv.promotionType();
        if (pt == ::chess::PieceType::QUEEN) {
            // Use queen-ray slot (forward for straight, diagonal for captures).
            const int slot = detail::queen_ray_slot(from, to);
            if (slot < 0) return -1;
            return from * kMovetypes + slot;
        } else {
            const int slot = detail::underpromo_slot(from, to, pt);
            if (slot < 0) return -1;
            return from * kMovetypes + slot;
        }
    }

    // NORMAL move — try queen-ray first, then knight
    {
        const int slot = detail::queen_ray_slot(from, to);
        if (slot >= 0) return from * kMovetypes + slot;
    }
    {
        const int slot = detail::knight_slot(from, to);
        if (slot >= 0) return from * kMovetypes + slot;
    }
    return -1;
}

/// Inverse of move_to_policy_idx.
/// Given a policy index and the current board, reconstruct the chess::Move.
/// Returns chess::Move::NO_MOVE if the index is out of range or invalid.
/// The returned move is not guaranteed to be legal; the caller should validate
/// via board.isLegal() if needed.
inline ::chess::Move policy_idx_to_move(int idx, const ::chess::Board& board) noexcept {
    if (idx < 0 || idx >= kPolicySize) return ::chess::Move::NO_MOVE;

    const int from_sq_idx = idx / kMovetypes;
    const int type_slot   = idx % kMovetypes;

    const ::chess::Square from_sq(from_sq_idx);
    const int from_file = from_sq_idx & 7;
    const int from_rank = from_sq_idx >> 3;

    // Queen-ray slots [0..55]
    if (type_slot < kQueenSlots) {
        const int dir  = type_slot / kMoveDistances;
        const int dist = type_slot % kMoveDistances + 1;
        const int to_file = from_file + kDirDeltaFile[dir] * dist;
        const int to_rank = from_rank + kDirDeltaRank[dir] * dist;
        if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) return ::chess::Move::NO_MOVE;
        const int to_sq_idx = to_rank * 8 + to_file;
        const ::chess::Square to_sq(to_sq_idx);

        // Check for promotion (pawn on rank 7 moving to rank 8, or rank 2 to rank 1)
        const auto piece = board.at(from_sq);
        if (piece.type() == ::chess::PieceType::PAWN) {
            const bool white_promo = (piece.color() == ::chess::Color::WHITE && from_rank == 6 && to_rank == 7);
            const bool black_promo = (piece.color() == ::chess::Color::BLACK && from_rank == 1 && to_rank == 0);
            if (white_promo || black_promo) {
                // Queen promotion uses the queen-ray slot
                return ::chess::Move::make<::chess::Move::PROMOTION>(from_sq, to_sq, ::chess::PieceType::QUEEN);
            }
        }

        // Check for castling (king moving more than 1 square along rank)
        if (piece.type() == ::chess::PieceType::KING && std::abs((to_sq_idx & 7) - from_file) > 1
            && from_rank == to_rank) {
            // Try to encode as castling — the actual rook square is what chess-library uses
            // for castling moves in standard (non-960) notation
            return ::chess::Move::make<::chess::Move::CASTLING>(from_sq, to_sq);
        }

        // Check for en passant
        const auto ep_sq = board.enpassantSq();
        if (piece.type() == ::chess::PieceType::PAWN && ep_sq != ::chess::Square::NO_SQ
            && to_sq == ep_sq) {
            return ::chess::Move::make<::chess::Move::ENPASSANT>(from_sq, to_sq);
        }

        return ::chess::Move::make<::chess::Move::NORMAL>(from_sq, to_sq);
    }

    // Knight slots [56..63]
    if (type_slot < kQueenSlots + kKnightSlots) {
        const int k = type_slot - kQueenSlots;
        const int to_file = from_file + kKnightDeltaFile[k];
        const int to_rank = from_rank + kKnightDeltaRank[k];
        if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) return ::chess::Move::NO_MOVE;
        const int to_sq_idx = to_rank * 8 + to_file;
        return ::chess::Move::make<::chess::Move::NORMAL>(from_sq, ::chess::Square(to_sq_idx));
    }

    // Underpromotion slots [64..72]
    {
        const int k   = type_slot - kQueenSlots - kKnightSlots;
        const int dir  = k / 3;
        const int piece_offset = k % 3;
        const int df = (dir == 0) ? -1 : (dir == 1) ? 0 : +1;
        const int to_file = from_file + df;
        // Determine promotion rank based on who's moving
        const auto piece = board.at(from_sq);
        int to_rank = -1;
        if (piece.color() == ::chess::Color::WHITE) to_rank = from_rank + 1;
        else if (piece.color() == ::chess::Color::BLACK) to_rank = from_rank - 1;
        else return ::chess::Move::NO_MOVE;

        if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) return ::chess::Move::NO_MOVE;
        const int to_sq_idx = to_rank * 8 + to_file;

        ::chess::PieceType pt;
        switch (piece_offset) {
            case 0: pt = ::chess::PieceType::KNIGHT; break;
            case 1: pt = ::chess::PieceType::BISHOP; break;
            case 2: pt = ::chess::PieceType::ROOK;   break;
            default: return ::chess::Move::NO_MOVE;
        }
        return ::chess::Move::make<::chess::Move::PROMOTION>(from_sq, ::chess::Square(to_sq_idx), pt);
    }
}

} // namespace chess_mlx::chess
