// engine/src/chess/chess_traits.cpp
//
// Out-of-line definitions for chess_mlx::chess::ChessTraits.
//
// Most of the implementation is in the headers (chess_traits.hpp and
// chess_encoding.hpp are effectively the full implementation since chess-library
// is header-only).  This translation unit exists to:
//   1. Satisfy the CMake chess_mlx_chess STATIC target.
//   2. Provide a stable linkage unit for any future non-inline symbols.

// chess-library is header-only — include it once here so we pay the compilation
// cost in a single TU rather than in every file that includes chess_traits.hpp.
// (chess_traits.hpp already guards against double-inclusion via #pragma once,
// but having one explicit TU is a common pattern for header-only libs.)

#include <chess/chess_traits.hpp>

namespace chess_mlx::chess {

// Ensure the chess-library STARTPOS constant is linked.
// This is a defensive measure; chess.hpp exposes it in namespace chess::constants.
static_assert(sizeof(::chess::constants::STARTPOS) > 0,
              "chess-library STARTPOS must be available");

} // namespace chess_mlx::chess
