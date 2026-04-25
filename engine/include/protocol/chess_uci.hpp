// SPDX-License-Identifier: MIT
// engine/include/protocol/chess_uci.hpp
//
// ChessUciTraits + Lc0ChessUciTraits + run_uci_loop_with_backend()
//
// Included only by chess_engine and the test harness.  Pulls in chess-library.

#pragma once

#include "chess/chess_traits.hpp"
#include "chess/lc0_chess_traits.hpp"
#include "chess/uci.hpp"
#include "nn/backend.hpp"
#include "protocol/uci_loop.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>

namespace chess_mlx::protocol {

// ---------------------------------------------------------------------------
// ChessUciTraits — hooks for chess, native 19-plane / 4672-slot.
// ---------------------------------------------------------------------------
struct ChessUciTraits {
    using Traits   = chess_mlx::chess::ChessTraits;
    using Position = Traits::Position;
    using Move     = Traits::Move;

    static std::string engine_id_name() { return "chess_mlx_engine"; }
    static std::string init_string()    { return "uci"; }
    static std::string ok_string()      { return "uciok"; }
    static std::string newgame_keyword(){ return "ucinewgame"; }

    static Position start_position()    { return Position{}; }

    static Position parse_position_command(std::string_view line) {
        if (line.substr(0, 9) == "position ") line.remove_prefix(9);
        return chess_mlx::chess::parse_position_command(line);
    }

    static std::string move_to_string(const Move& m) {
        return chess_mlx::chess::move_to_uci(m);
    }

    static bool stm_is_white(const Position& p) {
        return p.board.sideToMove() == ::chess::Color::WHITE;
    }

    static Move parse_move(const Position& pos, const std::string& uci_str) {
        return chess_mlx::chess::uci_to_move(pos, uci_str);
    }
};

// ---------------------------------------------------------------------------
// Lc0ChessUciTraits — hooks for chess, LC0 112-plane / 1858-slot.
// ---------------------------------------------------------------------------
struct Lc0ChessUciTraits {
    using Traits   = chess_mlx::chess::Lc0ChessTraits;
    using Position = Traits::Position;
    using Move     = Traits::Move;

    static std::string engine_id_name() { return "chess_mlx_lc0_engine"; }
    static std::string init_string()    { return "uci"; }
    static std::string ok_string()      { return "uciok"; }
    static std::string newgame_keyword(){ return "ucinewgame"; }

    static Position start_position()    { return Position{}; }

    static Position parse_position_command(std::string_view line) {
        if (line.substr(0, 9) == "position ") line.remove_prefix(9);
        return chess_mlx::chess::parse_position_command(line);
    }

    static std::string move_to_string(const Move& m) {
        return chess_mlx::chess::move_to_uci(m);
    }

    static bool stm_is_white(const Position& p) {
        return p.board.sideToMove() == ::chess::Color::WHITE;
    }

    static Move parse_move(const Position& pos, const std::string& uci_str) {
        return chess_mlx::chess::uci_to_move(pos, uci_str);
    }
};

// ---------------------------------------------------------------------------
// Convenience wrappers.
// ---------------------------------------------------------------------------

// Standard (19-plane / 4672-slot) version.
inline int run_uci_loop_with_backend(std::shared_ptr<nn::NNBackend> backend,
                                      int threads = 1,
                                      int multipv = 1) {
    using Loop = UciLoop<ChessUciTraits>;
    typename Loop::MCTS::Config cfg;
    cfg.threads     = std::max(1, threads);
    cfg.multipv     = std::max(1, multipv);
    cfg.input_size  = 64 * 19;
    cfg.policy_size = 4672;
    Loop loop(backend, cfg);
    return loop.run();
}

// LC0 (112-plane / 1858-slot) version.
inline int run_lc0_uci_loop_with_backend(std::shared_ptr<nn::NNBackend> backend,
                                          int threads = 1,
                                          int multipv = 1) {
    using Loop = UciLoop<Lc0ChessUciTraits>;
    typename Loop::MCTS::Config cfg;
    cfg.threads     = std::max(1, threads);
    cfg.multipv     = std::max(1, multipv);
    cfg.input_size  = 64 * 112;   // 7168
    cfg.policy_size = 1858;
    Loop loop(backend, cfg);
    return loop.run();
}

} // namespace chess_mlx::protocol
