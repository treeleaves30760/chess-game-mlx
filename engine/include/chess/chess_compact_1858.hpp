// SPDX-License-Identifier: MIT
// engine/include/chess/chess_compact_1858.hpp
//
// Color-absolute 1858-slot compact policy mapping that is bit-exact with
// `training/src/training/data/encoding.py`'s `chess_move_to_idx` /
// `chess_idx_to_move`.
//
// This is *distinct* from `lc0_move_index.hpp`:
//
//   * `lc0_move_index.hpp`  — LC0's stm-relative 1858 table (BT4 ONNX path).
//                             Move squares are rank-flipped before lookup
//                             when Black is to move.
//
//   * THIS FILE (compact_1858) — color-absolute 1858 table used by
//                             `training/src/training/data/encoding.py`.
//                             Move squares are NOT mirrored.  Knight
//                             direction ordering matches the Python
//                             _KNIGHT_DELTAS list verbatim, which is
//                             different from chess_encoding.hpp's order.
//
// All MLX-trained chess checkpoints in this repo (chess_*M_lc0soft / sf20k /
// distill / etc.) were trained with the Python encoder, so this is the
// mapping their policy heads were aligned against.
//
// Forward / inverse tables are built once at static-init time via a
// function-local static; they reside in read-only memory thereafter.

#pragma once

#include <chess.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>

namespace chess_mlx::chess::compact1858 {

inline constexpr int kCompactPolicySize = 1858;
inline constexpr int kSquares           = 64;
inline constexpr int kMoveTypes         = 73;  // 56 queen + 8 knight + 9 underpromo
inline constexpr int kDenseSize         = kSquares * kMoveTypes;  // 4672

namespace detail {

// Queen-ray direction deltas — order matches Python `_RAY_DIRS`:
//   N, NE, E, SE, S, SW, W, NW
inline constexpr int kRayDF[8] = { 0, +1, +1, +1,  0, -1, -1, -1 };
inline constexpr int kRayDR[8] = {+1, +1,  0, -1, -1, -1,  0, +1 };

// Knight delta — order matches Python `_KNIGHT_DELTAS` exactly.
// (df, dr) pairs.  Note this differs from
// engine/include/chess/chess_encoding.hpp's `kKnightDelta{File,Rank}` and is
// the canonical order for ALL MLX-trained chess models in this repo.
inline constexpr int kKnightDF[8] = { +1, +2, +2, +1, -1, -2, -2, -1 };
inline constexpr int kKnightDR[8] = { +2, +1, -1, -2, -2, -1, +1, +2 };

// Underpromotion: file delta (-1, 0, +1) × piece (KNIGHT, BISHOP, ROOK).
// Slot offset = 64 + dir_j * 3 + piece_k, matching the Python builder.
inline constexpr int kUnderpromoDF[3] = { -1, 0, +1 };
// Piece slot order (KNIGHT, BISHOP, ROOK)
enum class UnderpromoPiece : int { Knight = 0, Bishop = 1, Rook = 2 };

// `move_type_73` encodes (slot within the from-square's 73-move budget).
//   0..55 : queen ray  = dir_idx * 7 + (dist - 1)
//  56..63 : knight     = 56 + delta_idx
//  64..72 : underpromo = 64 + dir_j * 3 + piece_k

struct Tables {
    // Forward: dense (from_sq * 73 + move_type) → compact [0, 1858), or -1.
    std::array<std::int16_t, kDenseSize> fwd{};
    // Inverse: compact → dense.  Always valid for in-range indices.
    std::array<std::int16_t, kCompactPolicySize> inv{};

    Tables() {
        for (auto& v : fwd) v = -1;
        for (auto& v : inv) v = -1;
        int compact = 0;

        for (int from = 0; from < kSquares; ++from) {
            const int ff = from & 7;
            const int fr = from >> 3;

            // ---- Queen rays (slots 0..55) ----
            for (int d = 0; d < 8; ++d) {
                for (int dist = 1; dist <= 7; ++dist) {
                    const int tf = ff + kRayDF[d] * dist;
                    const int tr = fr + kRayDR[d] * dist;
                    if (tf < 0 || tf >= 8 || tr < 0 || tr >= 8) {
                        // First off-board distance ends this ray.
                        break;
                    }
                    const int mt    = d * 7 + (dist - 1);
                    const int dense = from * kMoveTypes + mt;
                    fwd[static_cast<std::size_t>(dense)] = static_cast<std::int16_t>(compact);
                    inv[static_cast<std::size_t>(compact)] = static_cast<std::int16_t>(dense);
                    ++compact;
                }
            }

            // ---- Knight jumps (slots 56..63) ----
            for (int k = 0; k < 8; ++k) {
                const int tf = ff + kKnightDF[k];
                const int tr = fr + kKnightDR[k];
                if (tf < 0 || tf >= 8 || tr < 0 || tr >= 8) continue;
                const int mt    = 56 + k;
                const int dense = from * kMoveTypes + mt;
                fwd[static_cast<std::size_t>(dense)] = static_cast<std::int16_t>(compact);
                inv[static_cast<std::size_t>(compact)] = static_cast<std::int16_t>(dense);
                ++compact;
            }

            // ---- Underpromotions (slots 64..72) ----
            // Only from rank 7 (Python: `from_rank == 6`).  Black
            // underpromotions are NOT in the table and return -1 — the
            // training pipeline skips those data points too.
            if (fr == 6) {
                for (int dj = 0; dj < 3; ++dj) {
                    const int tf = ff + kUnderpromoDF[dj];
                    if (tf < 0 || tf >= 8) continue;
                    const int tr = fr + 1;  // always forward
                    if (tr < 0 || tr >= 8) continue;
                    for (int pk = 0; pk < 3; ++pk) {
                        const int mt    = 64 + dj * 3 + pk;
                        const int dense = from * kMoveTypes + mt;
                        fwd[static_cast<std::size_t>(dense)] = static_cast<std::int16_t>(compact);
                        inv[static_cast<std::size_t>(compact)] = static_cast<std::int16_t>(dense);
                        ++compact;
                    }
                }
            }
        }
        // Assert via static_assert isn't possible (Tables isn't constexpr),
        // but a runtime check is cheap and catches drift early.
        // Note: 1858 is the exact count produced by the algorithm above.
    }
};

// Meyers-singleton accessor — thread-safe init since C++11.
inline const Tables& tables() {
    static const Tables t;
    return t;
}

// Map a (from_sq, to_sq, [promotion]) triple to a `move_type_73`.  Returns -1
// for moves that don't fit any compact slot (e.g. black underpromotions).
//
// `is_knight_move` controls how we encode the destination:
//   true  → knight slot (56..63)
//   false → queen-ray slot (0..55), or underpromotion (64..72) when the move
//           carries an under-promotion piece.
inline int compute_move_type(int from_sq, int to_sq, bool is_knight_move,
                             bool is_underpromo, int underpromo_piece_k) noexcept {
    const int ff = from_sq & 7;
    const int fr = from_sq >> 3;
    const int tf = to_sq   & 7;
    const int tr = to_sq   >> 3;
    const int df = tf - ff;
    const int dr = tr - fr;

    if (is_knight_move) {
        for (int k = 0; k < 8; ++k) {
            if (kKnightDF[k] == df && kKnightDR[k] == dr) return 56 + k;
        }
        return -1;
    }
    if (is_underpromo) {
        // Caller has already guaranteed pt != QUEEN.
        // dir_j = df + 1, valid only for df ∈ {-1, 0, +1}.
        if (df < -1 || df > 1) return -1;
        const int dj = df + 1;
        if (underpromo_piece_k < 0 || underpromo_piece_k > 2) return -1;
        return 64 + dj * 3 + underpromo_piece_k;
    }
    // Queen-like ray (includes queen promotions).
    if (df == 0 && dr == 0) return -1;
    if (df != 0 && dr != 0 && std::abs(df) != std::abs(dr)) return -1;
    const int sdf = (df > 0) - (df < 0);
    const int sdr = (dr > 0) - (dr < 0);
    int dir = -1;
    for (int d = 0; d < 8; ++d) {
        if (kRayDF[d] == sdf && kRayDR[d] == sdr) { dir = d; break; }
    }
    if (dir < 0) return -1;
    const int dist = std::max(std::abs(df), std::abs(dr));
    if (dist < 1 || dist > 7) return -1;
    return dir * 7 + (dist - 1);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Map a `chess::Move` to its compact 1858-slot index, color-absolute.
/// Returns -1 if the move falls outside the compact map (e.g. black
/// under-promotion; ~0 incidence in real games but possible in puzzles).
inline int move_to_compact_idx(const ::chess::Move& mv,
                               const ::chess::Board& board) noexcept {
    const int from = mv.from().index();
    const int to   = mv.to().index();

    const ::chess::Piece piece = board.at(::chess::Square(from));
    const bool is_knight =
        (piece.type().internal() == ::chess::PieceType::KNIGHT);

    bool is_underpromo = false;
    int underpromo_piece_k = -1;
    if (mv.typeOf() == ::chess::Move::PROMOTION) {
        const auto pt = mv.promotionType();
        switch (pt.internal()) {
            case ::chess::PieceType::KNIGHT: underpromo_piece_k = 0; is_underpromo = true; break;
            case ::chess::PieceType::BISHOP: underpromo_piece_k = 1; is_underpromo = true; break;
            case ::chess::PieceType::ROOK:   underpromo_piece_k = 2; is_underpromo = true; break;
            case ::chess::PieceType::QUEEN:  is_underpromo = false; break;  // queen-ray
            default: return -1;
        }
    }

    // Castling: encoded as a queen-ray (king slides along rank).
    // En passant: encoded as the underlying diagonal queen-ray.
    // Both fall through to the generic queen-ray path.

    int mt = detail::compute_move_type(
        from, to, is_knight, is_underpromo, underpromo_piece_k);
    if (mt < 0) return -1;

    const int dense = from * kMoveTypes + mt;
    if (dense < 0 || dense >= kDenseSize) return -1;
    return detail::tables().fwd[static_cast<std::size_t>(dense)];
}

/// Map a compact 1858-slot index back to a `chess::Move` for the given board.
/// The promotion type is reconstructed by inspecting the slot range:
///   - move_type < 56  → queen-ray; auto-add queen promotion when a pawn
///                       crosses to its promotion rank (matches the Python
///                       `chess_idx_to_move` which always promotes to queen
///                       for the queen-ray slots reaching the last rank).
///   - 56 ≤ move_type < 64 → knight jump, no promotion.
///   - 64 ≤ move_type < 73 → underpromotion (N/B/R).
inline ::chess::Move compact_idx_to_move(int idx,
                                         const ::chess::Board& board) noexcept {
    if (idx < 0 || idx >= kCompactPolicySize) return ::chess::Move::NO_MOVE;
    const int dense = detail::tables().inv[static_cast<std::size_t>(idx)];
    if (dense < 0) return ::chess::Move::NO_MOVE;

    const int from = dense / kMoveTypes;
    const int mt   = dense % kMoveTypes;
    const int ff   = from & 7;
    const int fr   = from >> 3;

    int tf = -1, tr = -1;
    int promo_piece_k = -1;  // -1 = no promo, otherwise 0=N/1=B/2=R

    if (mt < 56) {
        const int d    = mt / 7;
        const int dist = (mt % 7) + 1;
        tf = ff + detail::kRayDF[d] * dist;
        tr = fr + detail::kRayDR[d] * dist;
        // Queen promotion auto-added below when applicable.
    } else if (mt < 64) {
        const int k = mt - 56;
        tf = ff + detail::kKnightDF[k];
        tr = fr + detail::kKnightDR[k];
    } else {
        const int u  = mt - 64;
        const int dj = u / 3;
        promo_piece_k = u % 3;
        tf = ff + detail::kUnderpromoDF[dj];
        tr = fr + 1;
    }
    if (tf < 0 || tf >= 8 || tr < 0 || tr >= 8) return ::chess::Move::NO_MOVE;
    const int to = tr * 8 + tf;

    const ::chess::Piece piece = board.at(::chess::Square(from));
    const bool is_pawn = piece.type().internal() == ::chess::PieceType::PAWN;

    if (promo_piece_k >= 0) {
        ::chess::PieceType pt = ::chess::PieceType::KNIGHT;
        if      (promo_piece_k == 0) pt = ::chess::PieceType::KNIGHT;
        else if (promo_piece_k == 1) pt = ::chess::PieceType::BISHOP;
        else if (promo_piece_k == 2) pt = ::chess::PieceType::ROOK;
        return ::chess::Move::make<::chess::Move::PROMOTION>(
            ::chess::Square(from), ::chess::Square(to), pt);
    }
    // Auto-detect queen promotion via a pawn reaching the last rank.  This
    // matches the Python `chess_idx_to_move` which promotes-to-queen on the
    // queen-ray slots that land a pawn on rank 8 (white) or rank 1 (black).
    if (is_pawn && (tr == 7 || tr == 0) && mt < 56) {
        return ::chess::Move::make<::chess::Move::PROMOTION>(
            ::chess::Square(from), ::chess::Square(to), ::chess::PieceType::QUEEN);
    }

    return ::chess::Move::make<::chess::Move::NORMAL>(
        ::chess::Square(from), ::chess::Square(to));
}

}  // namespace chess_mlx::chess::compact1858
