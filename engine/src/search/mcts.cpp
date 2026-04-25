// SPDX-License-Identifier: MIT
// engine/src/search/mcts.cpp
//
// MCTS<Traits> is implemented entirely in the header (engine/include/search/mcts.hpp)
// so that chess and shogi can each instantiate it with their own Traits type.
// This TU exists only to satisfy the chess_mlx_search CMake target with at
// least one compiled translation unit.

namespace chess_mlx::search {
// Explicit anchor — keeps the archive non-empty.
void mcts_anchor() noexcept {}
} // namespace chess_mlx::search
