#pragma once

// ----------------------------------------------------------------------------
// engine/include/chess/uci.hpp
//
// Small helper wrappers around chess-library's built-in UCI utilities.
// These are the only chess-specific UCI parsing functions needed by the
// protocol layer; everything else is game-agnostic.
// ----------------------------------------------------------------------------

#include <chess.hpp>
#include <chess/chess_traits.hpp>

#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace chess_mlx::chess {

// ============================================================================
// FEN utilities
// ============================================================================

/// Apply a sequence of UCI move strings to a ChessPosition.
/// Each element of `moves` is a UCI move string (e.g. "e2e4", "e7e8q").
/// Throws std::invalid_argument if any move string is ill-formed or results
/// in an illegal move.
inline void apply_uci_moves(ChessPosition& pos, const std::vector<std::string>& moves) {
    for (const auto& uci_str : moves) {
        ::chess::Move m;
        try {
            m = ::chess::uci::uciToMove(pos.board, uci_str);
        } catch (const std::exception& e) {
            throw std::invalid_argument(
                std::string("apply_uci_moves: bad move '") + uci_str + "': " + e.what());
        }
        if (m == ::chess::Move::NO_MOVE) {
            throw std::invalid_argument(
                std::string("apply_uci_moves: illegal move '") + uci_str + "'");
        }
        pos.board.makeMove(m);
    }
}

/// Parse the `position` UCI command and return the resulting ChessPosition.
///
/// Accepted formats:
///   "startpos"
///   "startpos moves e2e4 e7e5 ..."
///   "fen <FEN>"
///   "fen <FEN> moves e2e4 e7e5 ..."
///
/// Throws std::invalid_argument on parse error.
inline ChessPosition parse_position_command(std::string_view cmd) {
    // Tokenise on whitespace
    std::vector<std::string_view> tokens;
    std::size_t i = 0;
    while (i < cmd.size()) {
        while (i < cmd.size() && cmd[i] == ' ') ++i;
        const std::size_t start = i;
        while (i < cmd.size() && cmd[i] != ' ') ++i;
        if (i > start) tokens.push_back(cmd.substr(start, i - start));
    }

    if (tokens.empty()) throw std::invalid_argument("parse_position_command: empty command");

    ChessPosition pos;

    std::size_t moves_offset = 0;

    if (tokens[0] == "startpos") {
        pos = ChessPosition();
        moves_offset = 1;
    } else if (tokens[0] == "fen") {
        // Reconstruct FEN (up to "moves" keyword or end)
        std::string fen_str;
        std::size_t t = 1;
        while (t < tokens.size() && tokens[t] != "moves") {
            if (!fen_str.empty()) fen_str += ' ';
            fen_str += tokens[t];
            ++t;
        }
        pos = ChessPosition(fen_str);
        moves_offset = t;
    } else {
        throw std::invalid_argument(
            std::string("parse_position_command: expected 'startpos' or 'fen', got '")
            + std::string(tokens[0]) + "'");
    }

    // Skip "moves" keyword
    if (moves_offset < tokens.size() && tokens[moves_offset] == "moves") {
        ++moves_offset;
    }

    // Apply move list
    for (std::size_t t = moves_offset; t < tokens.size(); ++t) {
        ::chess::Move m;
        try {
            m = ::chess::uci::uciToMove(pos.board, tokens[t]);
        } catch (const std::exception& e) {
            throw std::invalid_argument(
                std::string("parse_position_command: bad move '") + std::string(tokens[t]) + "': " + e.what());
        }
        if (m == ::chess::Move::NO_MOVE) {
            throw std::invalid_argument(
                std::string("parse_position_command: illegal move '") + std::string(tokens[t]) + "'");
        }
        pos.board.makeMove(m);
    }

    return pos;
}

/// Convert a ChessMove to its UCI string representation.
[[nodiscard]] inline std::string move_to_uci(const ChessMove& mv) {
    return ::chess::uci::moveToUci(mv.inner);
}

/// Parse a UCI move string into a ChessMove, given the current position.
/// Throws std::invalid_argument on bad input.
[[nodiscard]] inline ChessMove uci_to_move(const ChessPosition& pos, std::string_view uci_str) {
    ::chess::Move m;
    try {
        m = ::chess::uci::uciToMove(pos.board, uci_str);
    } catch (const std::exception& e) {
        throw std::invalid_argument(
            std::string("uci_to_move: '") + std::string(uci_str) + "': " + e.what());
    }
    if (m == ::chess::Move::NO_MOVE) {
        throw std::invalid_argument(
            std::string("uci_to_move: illegal move '") + std::string(uci_str) + "'");
    }
    return ChessMove(m);
}

} // namespace chess_mlx::chess
