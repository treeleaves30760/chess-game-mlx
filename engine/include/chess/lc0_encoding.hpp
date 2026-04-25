#pragma once

// ----------------------------------------------------------------------------
// engine/include/chess/lc0_encoding.hpp
//
// LC0-compatible 112-plane input encoder for chess positions.
//
// Algorithm matches LC0's src/neural/encoder.cc (GPL v3) —
// re-implemented from spec for license hygiene.
//
// Input tensor shape: [64 * 112] float32, row-major (plane-minor).
//   - Indexing: out[square * FEAT_DIM + plane] = value
//   - Square ordering: a1=0, b1=1, ..., h8=63  (identical to our 19-plane encoder)
//
// 112-plane layout:
//   Planes   0-103 : 8 history slots × 13 planes each
//     slot i (i=0 → current position, i=1 → 1 ply ago, ...):
//       plane i*13+0  : our  pawns
//       plane i*13+1  : our  knights
//       plane i*13+2  : our  bishops
//       plane i*13+3  : our  rooks
//       plane i*13+4  : our  queens
//       plane i*13+5  : our  kings
//       plane i*13+6  : their pawns
//       plane i*13+7  : their knights
//       plane i*13+8  : their bishops
//       plane i*13+9  : their rooks
//       plane i*13+10 : their queens
//       plane i*13+11 : their kings
//       plane i*13+12 : repetition flag (1.0 if position appeared ≥1 time before)
//   Planes 104-111 : auxiliary (all broadcast to all 64 squares)
//       104 : we can castle queenside  (from STM perspective)
//       105 : we can castle kingside
//       106 : they can castle queenside
//       107 : they can castle kingside
//       108 : side-to-move is BLACK  (1.0 if black to move, else 0.0)
//       109 : half-move clock / 100.0
//       110 : all zeros (was movecount; kept for network compat)
//       111 : all ones  (helps NN detect board edges)
//
// MIRRORING: LC0 always encodes from the STM's perspective.
//   When black is to move the board is rank-flipped and pieces are colour-swapped
//   before encoding. Square sq (0-63) in the output tensor corresponds to:
//     - a1..h8 (sq=0..63) when white is to move
//     - a8..h1 (sq→sq XOR 56 in normal coords) when black is to move
//   This means "our" pawns are always above the STM's starting rank, etc.
//
// History handling:
//   history[0] = position 1 ply before `pos` (most recent prior position)
//   history[1] = position 2 plies before, etc.
//   Missing history slots (past end of vector) are zero-padded (no piece planes).
//   The repetition plane for each history slot is computed by scanning backwards
//   through the provided positions (board hash equality = repetition).
//
// See also: encoding_spec.md §1 (19-plane native encoder)
// ----------------------------------------------------------------------------

#include <chess.hpp>
#include <chess/chess_traits.hpp>

#include <array>
#include <cstring>
#include <cstdint>
#include <vector>
#include <utility>

namespace chess_mlx::chess::lc0 {

// ============================================================================
// Constants
// ============================================================================

constexpr int SEQ_LEN       = 64;   ///< Number of squares per position
constexpr int FEAT_DIM      = 112;  ///< Total planes
constexpr int HISTORY_PLIES = 8;    ///< Number of history slots encoded
constexpr int PLANES_PER_BOARD = 13; ///< Planes per history slot
constexpr int AUX_PLANE_BASE = PLANES_PER_BOARD * HISTORY_PLIES; ///< = 104

// Auxiliary plane offsets (relative to AUX_PLANE_BASE = 104)
constexpr int kAuxWeCastle000   = 0; ///< We can castle queenside
constexpr int kAuxWeCastle00    = 1; ///< We can castle kingside
constexpr int kAuxTheyCastle000 = 2; ///< They can castle queenside
constexpr int kAuxTheyCastle00  = 3; ///< They can castle kingside
constexpr int kAuxSTM           = 4; ///< 1.0 if black to move
constexpr int kAuxRule50        = 5; ///< halfmove_clock / 100.0
constexpr int kAuxAllZeros      = 6; ///< All zeros (was movecount)
constexpr int kAuxAllOnes       = 7; ///< All ones

// ============================================================================
// Internal helpers
// ============================================================================

namespace detail {

/// Mirror a square index vertically (rank-flip).
/// sq = file + rank*8.  Mirror: new_rank = 7 - rank  =>  sq ^ 56.
inline constexpr int mirror_sq(int sq) noexcept { return sq ^ 56; }

/// Return the Bitboard (as uint64_t) for a given piece type and color
/// from our chess::Board.
///
/// The chess-library uses a1=bit0, h1=bit7, a2=bit8, ... same as LC0.
/// So no bit-level remapping is needed for white-to-move positions.
/// For black-to-move positions we rank-flip the entire uint64_t.
///
/// Rank-flip of a uint64_t: reverse byte order.
inline uint64_t flip_rank(uint64_t bb) noexcept {
    return __builtin_bswap64(bb);
}

/// Get the Bitboard for a piece type + color from our Board.
inline uint64_t get_bb(const ::chess::Board& board,
                        ::chess::PieceType pt,
                        ::chess::Color color) noexcept {
    return board.pieces(pt, color).getBits();
}


} // namespace detail

// ============================================================================
// Public API
// ============================================================================

/// Encode a position into LC0's 112-plane tensor.
///
/// Matches LC0's INPUT_CLASSICAL_112_PLANE encoding (src/neural/encoder.cc).
///
/// @param pos     Current position (most recent).
/// @param history Up to HISTORY_PLIES-1 prior positions, most-recent first.
///                history[0] = position immediately before pos (1 ply ago).
///                history[1] = 2 plies ago, etc.
/// @param out     Output buffer of at least SEQ_LEN * FEAT_DIM = 7168 floats.
///                Layout: out[sq * FEAT_DIM + plane].
/// @param fill_empty_history  When true (LC0's "fen_only" default): if history
///                is empty and the position is not standard startpos, fill
///                history slots 1-7 with the current position's planes (slot 0).
///                This matches LC0's HistoryFill::FEN_ONLY behavior.
inline void encode_nn(
    const ChessPosition& pos,
    const std::vector<ChessPosition>& history,
    float* out,
    bool fill_empty_history = true) noexcept
{
    // Zero the entire output buffer first.
    std::memset(out, 0, static_cast<std::size_t>(SEQ_LEN * FEAT_DIM) * sizeof(float));

    const ::chess::Board& board = pos.board;
    const bool black_to_move = (board.sideToMove() == ::chess::Color::BLACK);
    const ::chess::Color our_color   = board.sideToMove();
    const ::chess::Color their_color = ~our_color;

    // Piece type order: matches LC0's plane layout (P, N, B, R, Q, K).
    static constexpr ::chess::PieceType kPieceTypes[6] = {
        ::chess::PieceType::PAWN,
        ::chess::PieceType::KNIGHT,
        ::chess::PieceType::BISHOP,
        ::chess::PieceType::ROOK,
        ::chess::PieceType::QUEEN,
        ::chess::PieceType::KING
    };

    // -------------------------------------------------------------------------
    // History encoding: slot 0 = current pos, slot 1..7 = prior positions.
    //
    // LC0 encodes all slots from the current STM's perspective by applying a
    // "mirror" (rank-flip + color-swap) to each history board depending on a
    // cumulative flip flag. The flip flag toggles each time we step backwards
    // through REAL history (history_idx > 0 in LC0's notation). For padded
    // slots (no real board), the flip flag does NOT toggle, so all padded slots
    // use the same orientation as the last real slot.
    //
    // For a position with no history (FEN only, n_history=1):
    //   - slot 0: history_idx=0, flip_before_write=False
    //   - flip only toggles when history_idx > 0, never fires → flip stays False
    //   - all padded slots also use flip=False
    //   - should_mirror[slot] = we_are_black ^ flip_for_slot
    //     = we_are_black ^ False = we_are_black for all slots
    //
    // Consequence: for fen_only encoding (no real history), all 8 slots have
    // the same orientation and the same board → copy slot 0 into slots 1-7.
    //
    // For REAL history with n_history positions (n_history > 1):
    //   - slot i uses flip = (i % 2 == 1) for i < n_history
    //     (because the toggle fires for history_idx > 0 = for slots 0..n_history-2)
    //   - should_mirror[i] = we_are_black ^ (i % 2 == 1)
    //   - Color to extract as "ours" in the raw board:
    //       if should_mirror: their_color (before swap+flip = our side's pieces)
    //       else:             our_color
    //   - Rank-flip applies iff should_mirror.
    //
    // -------------------------------------------------------------------------

    // Helper: write one slot's 13 planes.
    // slot_board:    the physical board for this time step
    // ours_color:    which chess::Color in slot_board maps to "our" pieces
    // do_flip_rank:  whether to rank-flip all bitboards (matches should_mirror)
    // ep_undo_file:  if >= 0, undo the double pawn push on the "their" pawn BB.
    //                Applied AFTER rank-flip. Removes pawn from file+32, adds to
    //                file+48. Matches LC0's HistoryFill "en-passant undo" for
    //                padded history slots when the board carries an EP square.
    // repetition:    whether this position occurred before in the game
    // plane_base:    first plane index for this slot (slot * 13)
    auto write_slot = [&](const ::chess::Board& slot_board,
                          ::chess::Color ours_color,
                          bool do_flip_rank,
                          int ep_undo_file,
                          bool repetition,
                          int plane_base) noexcept {
        ::chess::Color theirs_color = ~ours_color;
        uint64_t bb_our[6], bb_their[6];
        for (int p = 0; p < 6; ++p) {
            bb_our[p]   = detail::get_bb(slot_board, kPieceTypes[p], ours_color);
            bb_their[p] = detail::get_bb(slot_board, kPieceTypes[p], theirs_color);
        }
        if (do_flip_rank) {
            for (int p = 0; p < 6; ++p) {
                bb_our[p]   = detail::flip_rank(bb_our[p]);
                bb_their[p] = detail::flip_rank(bb_their[p]);
            }
        }
        // EP undo: remove the double-pushed pawn from its current location (rank 4
        // in the representation, = file+32) and restore to original rank (rank 6,
        // = file+48). This applies only to padded history slots.
        if (ep_undo_file >= 0) {
            const unsigned f = static_cast<unsigned>(ep_undo_file);
            bb_their[0] &= ~(1ull << (f + 32u));  // remove from rank 4
            bb_their[0] |=  (1ull << (f + 48u));  // restore to rank 6
        }
        for (int p = 0; p < 6; ++p) {
            for (int sq = 0; sq < 64; ++sq) {
                out[sq * FEAT_DIM + plane_base + p]     = (bb_our[p]   >> sq) & 1u ? 1.0f : 0.0f;
                out[sq * FEAT_DIM + plane_base + p + 6] = (bb_their[p] >> sq) & 1u ? 1.0f : 0.0f;
            }
        }
        const float rep_val = repetition ? 1.0f : 0.0f;
        for (int sq = 0; sq < 64; ++sq) {
            out[sq * FEAT_DIM + plane_base + 12] = rep_val;
        }
    };

    // Number of real boards available: 1 (current) + history.size() (prior).
    const int n_real = 1 + static_cast<int>(history.size());

    // Detect startpos to skip fen_only fill (matching LC0's check against kStartposBoard).
    const bool is_startpos =
        (board == ::chess::Board(::chess::constants::STARTPOS));

    // Decide whether to fill empty history slots with the current board.
    const bool do_fen_fill =
        fill_empty_history && history.empty() && !is_startpos;

    // Compute EP undo file for padded slots.
    // When the position was constructed from a FEN with an EP square, padded
    // history slots undo the double pawn push: move "their" pawn from rank 4
    // back to rank 6. Matches LC0's HistoryFill "en-passant undo" trick.
    //
    // chess-library normalizes away EP squares when no legal EP capture exists.
    // ChessPosition::ep_file_hint preserves the raw FEN EP file (0..7, or -1).
    const int ep_undo_file = do_fen_fill ? pos.ep_file_hint : -1;

    for (int slot = 0; slot < HISTORY_PLIES; ++slot) {
        const int plane_base = slot * PLANES_PER_BOARD;

        // Determine the board for this slot.
        const ::chess::Board* slot_board = nullptr;
        bool is_padded = false;

        if (slot == 0) {
            slot_board = &board;
        } else if (static_cast<std::size_t>(slot - 1) < history.size()) {
            slot_board = &history[static_cast<std::size_t>(slot - 1)].board;
        } else if (do_fen_fill) {
            is_padded = true;
            slot_board = &board;  // same physical board as slot 0
        } else {
            continue;  // zero-padded (already done by memset)
        }

        // Compute flip_for_slot: matches Python's cumulative flip flag.
        // For real slots 0..n_real-1: flip_for_slot = (slot % 2 == 1).
        // For padded slots: flip stays at its value after slot n_real-1, which is
        // ((n_real-1) % 2 == 1). But the flag only increments when history_idx > 0;
        // the oldest real slot has history_idx=0 (no toggle). So:
        //   padded flip = flip value just after writing slot (n_real-1)
        //               = flip_for_slot(n_real-1) = ((n_real-1) % 2 == 1)
        // For n_real=1: padded flip = False. All padded slots share orientation with slot 0.
        const bool flip_for_slot = is_padded
            ? (((n_real - 1) % 2) == 1)
            : ((slot % 2) == 1);

        // should_mirror = we_are_black ^ flip_for_slot.
        // When True: rank-flip the bitboards (and color interpretation swaps).
        const bool should_mirror = black_to_move ^ flip_for_slot;

        // Color to extract as "ours" from the raw (unmirrored) board:
        // slot 0, 2, 4, 6 (even): same STM as current player → our_color
        // slot 1, 3, 5, 7 (odd): flipped STM → their_color
        // Padded slots all use the current board (same STM as slot 0) → our_color
        const bool slot_stm_is_ours = is_padded ? true : ((slot % 2) == 0);
        const ::chess::Color ours_color = slot_stm_is_ours ? our_color : their_color;

        // EP undo only for padded slots; not for slot 0 (which IS the current board,
        // correctly encoded with the real double-pushed pawn position).
        const int slot_ep_undo = (is_padded) ? ep_undo_file : -1;

        // Repetition detection (only for real boards with real history).
        bool repetition = false;
        if (!is_padded) {
            const uint64_t slot_hash = slot_board->hash();
            for (int j = slot + 2; j < HISTORY_PLIES; j += 2) {
                const ::chess::Board* prev_board = nullptr;
                if (j == 0) {
                    prev_board = &board;
                } else if (static_cast<std::size_t>(j - 1) < history.size()) {
                    prev_board = &history[static_cast<std::size_t>(j - 1)].board;
                }
                if (prev_board && prev_board->hash() == slot_hash) {
                    repetition = true;
                    break;
                }
            }
        }

        write_slot(*slot_board, ours_color, should_mirror, slot_ep_undo, repetition, plane_base);
    }

    // -------------------------------------------------------------------------
    // Auxiliary planes (104-111) — encoded from CURRENT position only.
    // -------------------------------------------------------------------------

    const auto cr = board.castlingRights();
    using CastleSide = ::chess::Board::CastlingRights::Side;

    const bool we_can_000   = cr.has(our_color,   CastleSide::QUEEN_SIDE);
    const bool we_can_00    = cr.has(our_color,   CastleSide::KING_SIDE);
    const bool they_can_000 = cr.has(their_color, CastleSide::QUEEN_SIDE);
    const bool they_can_00  = cr.has(their_color, CastleSide::KING_SIDE);

    const float f_we000   = we_can_000   ? 1.0f : 0.0f;
    const float f_we00    = we_can_00    ? 1.0f : 0.0f;
    const float f_they000 = they_can_000 ? 1.0f : 0.0f;
    const float f_they00  = they_can_00  ? 1.0f : 0.0f;
    const float f_stm     = black_to_move ? 1.0f : 0.0f;
    // Rule-50: raw half-move clock (LC0's convention — NOT divided by 100).
    const float f_rule50  = static_cast<float>(board.halfMoveClock());
    // plane AUX+6 = all zeros (already from memset)
    // plane AUX+7 = all ones

    for (int sq = 0; sq < 64; ++sq) {
        float* base = out + sq * FEAT_DIM + AUX_PLANE_BASE;
        base[kAuxWeCastle000]   = f_we000;
        base[kAuxWeCastle00]    = f_we00;
        base[kAuxTheyCastle000] = f_they000;
        base[kAuxTheyCastle00]  = f_they00;
        base[kAuxSTM]           = f_stm;
        base[kAuxRule50]        = f_rule50;
        base[kAuxAllZeros]      = 0.0f; // explicit (already 0 from memset)
        base[kAuxAllOnes]       = 1.0f;
    }
}

/// Convenience: encode a batch of (pos, history) pairs.
/// Output layout: [B, 64, 112] row-major (batch-major).
///
/// @param items  Vector of (current_pos, history) pairs.
/// @param out    Buffer of at least items.size() * SEQ_LEN * FEAT_DIM floats.
inline void encode_nn_batch(
    const std::vector<std::pair<ChessPosition, std::vector<ChessPosition>>>& items,
    float* out) noexcept
{
    constexpr std::size_t stride = static_cast<std::size_t>(SEQ_LEN * FEAT_DIM);
    for (std::size_t i = 0; i < items.size(); ++i) {
        encode_nn(items[i].first, items[i].second, out + i * stride);
    }
}

} // namespace chess_mlx::chess::lc0
