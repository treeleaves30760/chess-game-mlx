#pragma once

// This header provides lightweight move-type concepts shared across chess and shogi Traits.
// Each Traits specialisation defines its own concrete Move wrapper; this header provides
// the common vocabulary that the GameRules template depends on.
//
// A conforming Move type must be:
//   - Default-constructible (represents a null/no-move)
//   - Copyable and movable
//   - Equality-comparable
//   - Convertible to a string (UCI/USI notation) via a free function in the Traits namespace
//
// Concrete types:
//   - chess::ChessMove  — wraps ::chess::Move from Disservin/chess-library
//   - shogi::ShogiMove  — wraps cshogi move (implemented by shogi agent, Phase 2)

namespace chess_mlx {

/// Sentinel constant returned when no move is available (terminal position, etc.)
constexpr int POLICY_IDX_NONE = -1;

/// Maximum legal moves in any chess or shogi position (generous upper bound for array sizing).
constexpr int MAX_LEGAL_MOVES = 256;

} // namespace chess_mlx
