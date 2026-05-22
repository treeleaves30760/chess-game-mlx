// SPDX-License-Identifier: MIT
// engine/include/chess/chess_hybrid_traits.hpp
//
// ChessHybridTraits = native 19-plane input encoder + color-absolute 1858-slot
// compact policy mapping.
//
// This is the trait class for *every* MLX-trained chess checkpoint shipped in
// `checkpoints/`.  The Python training pipeline encodes positions as 64×19
// tensors (matching ChessTraits) and labels policy targets in a 1858-slot
// compact space built by `training/src/training/data/encoding.py` — color-
// absolute, with Python's knight-delta ordering.
//
// Why a third traits class:
//   * `ChessTraits`         — 19-plane encoder + 4672-slot policy.  No trained
//                             model uses this output space.
//   * `Lc0ChessTraits`      — 112-plane encoder + LC0's stm-relative 1858 table.
//                             Only the BT4 ONNX import lives here.
//   * `ChessHybridTraits`   — 19-plane encoder + color-absolute compact 1858.
//                             What the trained MLX checkpoints actually expect.
//
// Pre-ChessHybrid the engine plumbed every MLX checkpoint through
// ChessTraits, meaning every MCTS leaf's NN policy was read at indices that
// didn't correspond to the model's training-time labels — priors collapsed
// to near-uniform and the engine reverted to value-head-only play.

#pragma once

#include "chess/chess_traits.hpp"          // ChessTraits — encoder + Move/Position
#include "chess/chess_compact_1858.hpp"    // compact1858::move_to_compact_idx ...
#include "core/game_rules.hpp"

namespace chess_mlx::chess {

struct ChessHybridTraits {
    using Position = ChessPosition;
    using Move     = ChessMove;
    using MoveList = core::MoveList<ChessMove>;

    static constexpr const char* kEngineName = "chess_engine_hybrid";

    // --- Move generation / make / undo / terminal / hash : reuse ChessTraits ---
    static void generate_legal(const Position& pos, MoveList& out) {
        ChessTraits::generate_legal(pos, out);
    }
    static void apply(Position& pos, Move mv) noexcept { ChessTraits::apply(pos, mv); }
    static void undo (Position& pos, Move mv) noexcept { ChessTraits::undo(pos, mv); }
    static bool is_terminal(const Position& pos) noexcept {
        return ChessTraits::is_terminal(pos);
    }
    static float terminal_value(const Position& pos) noexcept {
        return ChessTraits::terminal_value(pos);
    }
    static std::uint64_t hash(const Position& pos) noexcept {
        return ChessTraits::hash(pos);
    }

    // --- Encoder : native 19-plane (color-absolute, with STM scalar plane) ---
    static void encode_nn(const Position& pos, float* out) noexcept {
        ChessTraits::encode_nn(pos, out);
    }

    // --- Policy mapping : color-absolute compact 1858 ---
    static int move_to_policy_idx(Move mv, const Position& pos) noexcept {
        return compact1858::move_to_compact_idx(mv.inner, pos.board);
    }
    static Move policy_idx_to_move(int idx, const Position& pos) noexcept {
        return ChessMove(compact1858::compact_idx_to_move(idx, pos.board));
    }

    // --- String / null helpers ---
    static std::string move_to_string(const Move& mv) {
        return ChessTraits::move_to_string(mv);
    }
    static bool is_null_move(const Move& mv) {
        return ChessTraits::is_null_move(mv);
    }
};

}  // namespace chess_mlx::chess
