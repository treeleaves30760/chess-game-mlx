#pragma once
// ----------------------------------------------------------------------------
// engine/include/chess/lc0_chess_traits.hpp
//
// Lc0ChessTraits — a drop-in replacement for ChessTraits that uses:
//   * LC0's 112-plane input encoder  (from lc0_encoding.hpp)
//   * LC0's 1858-slot policy space   (from lc0_move_index.hpp)
//
// MCTS stores LC0 slot indices directly in nodes.  The Lc0Backend outputs
// 1858-element policy vectors with one float per LC0 slot.
//
// This avoids the 1858→4672 remapping in the backend and keeps the node
// move_policy_idx in LC0 space throughout.  The UCI protocol layer simply
// converts slot→move when reporting bestmove.
// ----------------------------------------------------------------------------

#include "chess/chess_traits.hpp"         // for ChessPosition, ChessMove
#include "chess/lc0_encoding.hpp"         // lc0::encode_nn
#include "chess/lc0_move_index.hpp"       // lc0::lc0_move_to_slot, etc.
#include "core/game_rules.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace chess_mlx::chess {

// ============================================================================
// Lc0ChessTraits
// ============================================================================

struct Lc0ChessTraits {
    // Re-use the same Position and Move types as ChessTraits.
    using Position = ChessPosition;
    using Move     = ChessMove;
    using MoveList = core::MoveList<ChessMove>;

    static constexpr const char* kEngineName = "chess_engine_lc0";

    // -----------------------------------------------------------------------
    // Move generation (identical to ChessTraits)
    // -----------------------------------------------------------------------

    static void generate_legal(const Position& pos, MoveList& out) {
        ::chess::Movelist ml;
        ::chess::movegen::legalmoves(ml, pos.board);
        out.clear();
        for (const auto& m : ml) {
            out.push(ChessMove(m));
        }
    }

    static void apply(Position& pos, Move mv) noexcept {
        pos.board.makeMove(mv.inner);
    }

    static void undo(Position& pos, Move mv) noexcept {
        pos.board.unmakeMove(mv.inner);
    }

    // -----------------------------------------------------------------------
    // Terminal detection (identical to ChessTraits)
    // -----------------------------------------------------------------------

    static bool is_terminal(const Position& pos) noexcept {
        const auto [reason, result] = pos.board.isGameOver();
        return result != ::chess::GameResult::NONE;
    }

    static float terminal_value(const Position& pos) noexcept {
        const auto [reason, result] = pos.board.isGameOver();
        switch (result) {
            case ::chess::GameResult::WIN:  return +1.0f;
            case ::chess::GameResult::LOSE: return -1.0f;
            case ::chess::GameResult::DRAW: return  0.0f;
            default:                        return  0.0f;
        }
    }

    // -----------------------------------------------------------------------
    // Hashing (identical to ChessTraits)
    // -----------------------------------------------------------------------

    static std::uint64_t hash(const Position& pos) noexcept {
        return pos.board.hash();
    }

    // -----------------------------------------------------------------------
    // NN encoding — LC0's 112-plane scheme, no history (zero-padded).
    // Input tensor size: 64 * 112 = 7168 floats.
    // -----------------------------------------------------------------------

    static void encode_nn(const Position& pos, float* out) noexcept {
        static const std::vector<ChessPosition> kNoHistory;
        lc0::encode_nn(pos, kNoHistory, out);
    }

    // -----------------------------------------------------------------------
    // Policy mapping — LC0's 1858-slot space.
    //
    // move_to_policy_idx   : returns LC0 slot [0..1857], or -1.
    // policy_idx_to_move   : returns the ChessMove for a given LC0 slot.
    // -----------------------------------------------------------------------

    static int move_to_policy_idx(Move mv, const Position& pos) noexcept {
        const bool black_stm = (pos.board.sideToMove() == ::chess::Color::BLACK);
        return lc0::lc0_move_to_slot(mv.inner, black_stm);
    }

    static Move policy_idx_to_move(int idx, const Position& pos) noexcept {
        const bool black_stm = (pos.board.sideToMove() == ::chess::Color::BLACK);
        return ChessMove(lc0::lc0_slot_to_chess_move(idx, pos.board, black_stm));
    }

    // -----------------------------------------------------------------------
    // String conversion
    // -----------------------------------------------------------------------

    static std::string move_to_string(const Move& mv) {
        return mv.to_uci();
    }

    static bool is_null_move(const Move& mv) { return mv.is_null(); }
};

} // namespace chess_mlx::chess
