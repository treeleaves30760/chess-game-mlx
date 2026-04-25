#pragma once

// ----------------------------------------------------------------------------
// engine/include/chess/chess_traits.hpp
//
// ChessTraits — the concrete Traits type that plugs into GameRules<ChessTraits>.
// Delegates move-generation and board state to Disservin/chess-library.
//
// Public surface:
//   chess_mlx::chess::ChessTraits — the Traits tag struct
//   chess_mlx::chess::ChessPosition — wraps ::chess::Board (+ undo stack managed by Board)
//   chess_mlx::chess::ChessMove    — thin wrapper around ::chess::Move
// ----------------------------------------------------------------------------

#include <chess.hpp>
#include <core/game_rules.hpp>
#include <chess/chess_encoding.hpp>

#include <string>
#include <vector>
#include <cassert>
#include <cstdint>

namespace chess_mlx::chess {

// ============================================================================
// ChessMove — thin wrapper around ::chess::Move
// ============================================================================

struct ChessMove {
    ::chess::Move inner;

    ChessMove() noexcept : inner(::chess::Move::NO_MOVE) {}
    explicit ChessMove(::chess::Move m) noexcept : inner(m) {}

    [[nodiscard]] bool is_null() const noexcept {
        return inner == ::chess::Move::NO_MOVE;
    }

    bool operator==(const ChessMove& other) const noexcept {
        return inner == other.inner;
    }
    bool operator!=(const ChessMove& other) const noexcept {
        return inner != other.inner;
    }

    /// Returns UCI string representation, e.g. "e2e4", "e7e8q".
    [[nodiscard]] std::string to_uci() const {
        return ::chess::uci::moveToUci(inner);
    }
};

// ============================================================================
// ChessPosition — wraps ::chess::Board
// ============================================================================
//
// chess-library's Board already maintains an internal undo history via
// unmakeMove(), so we simply forward to it.
//
// ep_file_hint: the file index (0=a, …, 7=h) of the en-passant target square
// from the FEN, or -1 if absent. chess-library strips the EP square when no
// legal EP capture exists; this field preserves it for LC0 encoding.

struct ChessPosition {
    ::chess::Board board;
    int ep_file_hint = -1;  ///< Raw EP file from FEN; -1 if none.

    /// Default-constructs from the standard starting position.
    ChessPosition() { board.setFen(::chess::constants::STARTPOS); }

    /// Constructs from a FEN string, preserving the raw EP file.
    explicit ChessPosition(std::string_view fen) {
        board.setFen(fen);
        ep_file_hint = parse_ep_file(fen);
    }

    /// Returns the current FEN.
    [[nodiscard]] std::string fen() const { return board.getFen(); }

    /// Update ep_file_hint after makeMove/unmakeMove (not needed for FEN-only use).
    void update_ep_hint_from_board() {
        // After makeMove, the board's internal ep_sq_ is set correctly even if
        // no capture is legal (the board stores it for move-generation purposes).
        // However, chess-library may normalize it away. We trust the board.
        const ::chess::Square ep = board.enpassantSq();
        ep_file_hint = (ep.index() >= 0 && ep.index() < 64)
            ? static_cast<int>(ep.file())
            : -1;
    }

private:
    /// Parse the EP field from a FEN string. Returns file index (0..7) or -1.
    static int parse_ep_file(std::string_view fen) noexcept {
        // FEN format: pieces stm castling ep [half full]
        // Skip 3 space-delimited tokens to reach the EP field.
        std::size_t pos = 0;
        for (int tok = 0; tok < 3; ++tok) {
            pos = fen.find(' ', pos);
            if (pos == std::string_view::npos) return -1;
            ++pos;  // skip the space
        }
        if (pos < fen.size() && fen[pos] >= 'a' && fen[pos] <= 'h') {
            return static_cast<int>(fen[pos] - 'a');
        }
        return -1;
    }
};

// ============================================================================
// ChessTraits
// ============================================================================

struct ChessTraits {
    using Position = ChessPosition;
    using Move     = ChessMove;
    using MoveList = core::MoveList<ChessMove>;

    static constexpr const char* kEngineName = "chess_engine";

    // -------------------------------------------------------------------------
    // Move generation
    // -------------------------------------------------------------------------

    static void generate_legal(const Position& pos, MoveList& out) {
        ::chess::Movelist ml;
        ::chess::movegen::legalmoves(ml, pos.board);
        out.clear();
        for (const auto& m : ml) {
            out.push(ChessMove(m));
        }
    }

    // -------------------------------------------------------------------------
    // Make / undo
    // -------------------------------------------------------------------------

    static void apply(Position& pos, Move mv) noexcept {
        pos.board.makeMove(mv.inner);
    }

    static void undo(Position& pos, Move mv) noexcept {
        pos.board.unmakeMove(mv.inner);
    }

    // -------------------------------------------------------------------------
    // Terminal detection
    // -------------------------------------------------------------------------

    static bool is_terminal(const Position& pos) noexcept {
        const auto [reason, result] = pos.board.isGameOver();
        return result != ::chess::GameResult::NONE;
    }

    /// Returns the terminal value from the SIDE-TO-MOVE perspective (stm-POV).
    ///   +1 : side-to-move wins (cannot happen in standard chess but included)
    ///    0 : draw
    ///   -1 : side-to-move loses (checkmated)
    ///
    /// This is the convention used by MCTS for backpropagation.  If callers
    /// need white-POV, they can flip based on board.sideToMove().
    static float terminal_value(const Position& pos) noexcept {
        const auto [reason, result] = pos.board.isGameOver();
        switch (result) {
            case ::chess::GameResult::WIN:  return +1.0f; // stm wins
            case ::chess::GameResult::LOSE: return -1.0f; // stm loses (checkmated)
            case ::chess::GameResult::DRAW: return  0.0f;
            default:                        return  0.0f;
        }
    }

    // -------------------------------------------------------------------------
    // Hashing
    // -------------------------------------------------------------------------

    static std::uint64_t hash(const Position& pos) noexcept {
        return pos.board.hash();
    }

    // -------------------------------------------------------------------------
    // NN encoding
    // -------------------------------------------------------------------------

    static void encode_nn(const Position& pos, float* out) noexcept {
        chess_mlx::chess::encode_nn(pos.board, out);
    }

    // -------------------------------------------------------------------------
    // Policy mapping
    // -------------------------------------------------------------------------

    static int move_to_policy_idx(Move mv, const Position& /*pos*/) noexcept {
        return chess_mlx::chess::move_to_policy_idx(mv.inner);
    }

    static Move policy_idx_to_move(int idx, const Position& pos) noexcept {
        return ChessMove(chess_mlx::chess::policy_idx_to_move(idx, pos.board));
    }

    // -------------------------------------------------------------------------
    // String conversion (for protocol / multi-ponder notifications)
    // -------------------------------------------------------------------------

    static std::string move_to_string(const Move& mv) {
        return mv.to_uci();
    }

    // -------------------------------------------------------------------------
    // Null-move check
    // -------------------------------------------------------------------------
    static bool is_null_move(const Move& mv) { return mv.is_null(); }
};

} // namespace chess_mlx::chess
