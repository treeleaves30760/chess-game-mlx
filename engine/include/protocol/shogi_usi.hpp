// SPDX-License-Identifier: GPL-3.0-or-later
// engine/include/protocol/shogi_usi.hpp
//
// ShogiUsiTraits + run_usi_loop_with_backend() — included only by shogi_engine.
// This header pulls in cshogi (GPL v3); do NOT include from chess_engine.

#pragma once

#include "shogi/shogi_traits.hpp"
#include "shogi/usi.hpp"
#include "nn/backend.hpp"
#include "protocol/uci_loop.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>

namespace chess_mlx::protocol {

// ---------------------------------------------------------------------------
// ShogiUsiTraits — hooks for shogi.
// ---------------------------------------------------------------------------
struct ShogiUsiTraits {
    using Traits   = ::shogi::ShogiTraits;
    using Position = Traits::Position;
    using Move     = Traits::Move;

    static std::string engine_id_name() { return "shogi_mlx_engine"; }
    static std::string init_string()    { return "usi"; }
    static std::string ok_string()      { return "usiok"; }
    static std::string newgame_keyword(){ return "usinewgame"; }

    static Position start_position() {
        return ::shogi::ShogiPosition(::shogi::usi::start_sfen());
    }

    static Position parse_position_command(std::string_view line) {
        return ::shogi::usi::position_from_usi_command(std::string(line));
    }

    static std::string move_to_string(const Move& m) {
        return m.to_usi();
    }

    static bool stm_is_white(const Position& p) {
        // ShogiPosition::black_to_move() == true means 先手's turn which
        // we treat as the "white" (first-player) side for score-sign purposes.
        return p.black_to_move();
    }

    // Parse a USI move string in the context of the given position.
    static Move parse_move(const Position& pos, const std::string& usi_str) {
        return ::shogi::usi::move_from_usi(pos, usi_str);
    }
};

// Convenience wrapper — instantiate + run UciLoop<ShogiUsiTraits>.
inline int run_usi_loop_with_backend(std::shared_ptr<nn::NNBackend> backend,
                                      int threads = 1,
                                      int multipv = 1) {
    using Loop = UciLoop<ShogiUsiTraits>;
    typename Loop::MCTS::Config cfg;
    cfg.threads     = std::max(1, threads);
    cfg.multipv     = std::max(1, multipv);
    cfg.input_size  = 81 * 90;
    cfg.policy_size = 2187;
    Loop loop(backend, cfg);
    return loop.run();
}

} // namespace chess_mlx::protocol
