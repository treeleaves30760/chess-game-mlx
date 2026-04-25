// SPDX-License-Identifier: MIT
// engine/include/core/game_rules.hpp
//
// GameRules<Traits> — the shared abstraction over chess and shogi.
// Agent A writes the contract; each game's traits specialises it.
//
// IMPORTANT: This header is MIT-licensed and must NEVER include cshogi
// headers.  Keep it free of game-specific types so it can be safely
// included from chess_engine (which is purely MIT).

#pragma once

#include <cstdint>
#include <vector>

namespace core {

// ---------------------------------------------------------------------------
// MoveList — a thin, owning list of moves used by generate_legal().
// Each Traits type defines its own Move typedef; MoveList is parameterised
// on that type so the concrete type is erased here.
// ---------------------------------------------------------------------------
template <typename MoveT>
struct MoveList {
    std::vector<MoveT> moves;

    void clear()                       { moves.clear(); }
    void push(const MoveT& m)          { moves.push_back(m); }
    std::size_t size()           const { return moves.size(); }
    bool        empty()          const { return moves.empty(); }
    const MoveT& operator[](std::size_t i) const { return moves[i]; }
    MoveT&       operator[](std::size_t i)       { return moves[i]; }

    auto begin() const { return moves.begin(); }
    auto end()   const { return moves.end(); }
    auto begin()       { return moves.begin(); }
    auto end()         { return moves.end(); }
};

// ---------------------------------------------------------------------------
// GameRules<Traits>
//
// Each Traits type must provide:
//   using Position = ...;
//   using Move     = ...;
//
// And a static "engine tag" for display:
//   static constexpr const char* kEngineName;
//
// The static functions below are implemented by each Traits specialisation.
// ---------------------------------------------------------------------------
template <typename Traits>
struct GameRules {
    using Position = typename Traits::Position;
    using Move     = typename Traits::Move;
    using Moves    = MoveList<Move>;

    // -----------------------------------------------------------------------
    // Move generation
    // -----------------------------------------------------------------------
    // Fills `out` with all legal moves from `pos`.
    static void generate_legal(const Position& pos, Moves& out) {
        Traits::generate_legal(pos, out);
    }

    // -----------------------------------------------------------------------
    // Make / undo
    // -----------------------------------------------------------------------
    // Apply `m` to `pos` in-place.  The caller must ensure `m` is legal.
    static void apply(Position& pos, Move m) {
        Traits::apply(pos, m);
    }

    // Undo the last applied move.  The Traits implementation must maintain
    // the necessary history internally (e.g. cshogi's StateInfo stack).
    static void undo(Position& pos, Move m) {
        Traits::undo(pos, m);
    }

    // -----------------------------------------------------------------------
    // Terminal detection
    // -----------------------------------------------------------------------
    // Returns true if `pos` is a terminal state (no legal moves, draw, etc.)
    static bool is_terminal(const Position& pos) {
        return Traits::is_terminal(pos);
    }

    // Returns the terminal value from the perspective of the side-to-move.
    // {-1, 0, +1} — positive = current player wins.
    // Only valid when is_terminal() is true.
    static float terminal_value(const Position& pos) {
        return Traits::terminal_value(pos);
    }

    // -----------------------------------------------------------------------
    // Hashing
    // -----------------------------------------------------------------------
    // Returns a Zobrist (or equivalent) hash of `pos`.
    static std::uint64_t hash(const Position& pos) {
        return Traits::hash(pos);
    }

    // -----------------------------------------------------------------------
    // Neural-network encoding
    // -----------------------------------------------------------------------
    // Writes the flat float32 tensor into `out`.
    // `out` must point to at least (num_squares × num_channels) floats.
    // Layout: [square_0_channel_0, square_0_channel_1, ..., square_N_channel_C]
    // i.e. the square is the slow index (row-major [squares, channels]).
    static void encode_nn(const Position& pos, float* out) {
        Traits::encode_nn(pos, out);
    }

    // -----------------------------------------------------------------------
    // Policy mapping
    // -----------------------------------------------------------------------
    // Maps a legal move to a flat policy index in [0, policy_size).
    // Returns -1 if the move has no valid encoding (should not happen for
    // legal moves — treat as an assertion failure in debug builds).
    static int move_to_policy_idx(Move m, const Position& pos) {
        return Traits::move_to_policy_idx(m, pos);
    }

    // Maps a policy index back to a Move.
    // The returned move may be illegal; the caller must validate if needed.
    static Move policy_idx_to_move(int idx, const Position& pos) {
        return Traits::policy_idx_to_move(idx, pos);
    }
};

} // namespace core
