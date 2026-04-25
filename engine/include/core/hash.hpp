#pragma once

#include <cstdint>

namespace chess_mlx {

/// 64-bit Zobrist key type alias.
/// Used throughout the engine for position hashing and transposition table keys.
using ZobristKey = std::uint64_t;

} // namespace chess_mlx
