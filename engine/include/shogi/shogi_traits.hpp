// SPDX-License-Identifier: GPL-3.0-or-later
// engine/include/shogi/shogi_traits.hpp
//
// ShogiTraits — specialises the GameRules<Traits> contract for shogi.
//
// LICENSE NOTE: This file includes cshogi headers (derived from Apery /
// Stockfish) which are licensed under GPL v3.  Therefore this file and any
// translation unit that includes it are also subject to GPL v3.  The chess
// engine binary (chess_engine) must NEVER include this header.
//
// cshogi upstream: engine/third_party/cshogi-upstream/src/
// Original authors: Tord Romstad, Marco Costalba, Joona Kiiski, Gary Linscott,
//                   Hiraoka Takuya, TadaoYamaoka.

#pragma once

// ---- cshogi headers (GPL) -------------------------------------------------
// These are cshogi upstream headers found via the cshogi_core include path
// (engine/third_party/cshogi-upstream/src/).  They do NOT use our shogi/
// wrapper headers.
#include "position.hpp"
#include "generateMoves.hpp"
#include "move.hpp"
#include "hand.hpp"
#include "piece.hpp"
#include "square.hpp"
#include "init.hpp"
// Note: we do NOT include cshogi's "usi.hpp" here to avoid name collision
// with our own engine/include/shogi/usi.hpp.  The DefaultStartPositionSFEN
// constant is included via the .cpp file instead.
// ---------------------------------------------------------------------------

#include "core/game_rules.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>

namespace shogi {

// ---------------------------------------------------------------------------
// ShogiRuntime — idempotent, thread-safe initialisation of cshogi globals.
// Must be called once before any Position or move-gen usage.
// ---------------------------------------------------------------------------
struct ShogiRuntime {
    static void ensure_initialized();
private:
    static std::once_flag s_init_flag;
};

// ---------------------------------------------------------------------------
// ShogiMove — thin wrapper hiding cshogi's Move from downstream consumers.
//
// Only files that include shogi_traits.hpp (and therefore accept GPL) may
// inspect the raw cshogi Move.  All other code operates via ShogiMove's
// opaque interface.
// ---------------------------------------------------------------------------
struct ShogiMove {
    // Default: MoveNone
    ShogiMove() : raw_(::Move::moveNone()) {}

    // Explicit construction from a cshogi Move.
    explicit ShogiMove(::Move m) : raw_(m) {}

    // Access the underlying cshogi Move (only available in GPL-licensed code).
    ::Move raw() const { return raw_; }

    bool is_drop()      const { return raw_.isDrop(); }
    bool is_promotion() const { return raw_.isPromotion() != 0; }
    bool is_none()      const { return raw_.value() == ::Move::MoveNone; }

    // USI move string (e.g. "7g7f", "P*5f")
    std::string to_usi() const { return raw_.toUSI(); }

    // Equality compares the logical move only: (from, to, promote).
    // cshogi stores pieceTypeFrom (bits 16-19) and captured piece (bits 20-23)
    // in the full value, but these are position-dependent metadata not part of
    // the move identity.  We use proFromAndTo() = bits 0-14 for comparison.
    bool operator==(const ShogiMove& o) const {
        return raw_.proFromAndTo() == o.raw_.proFromAndTo();
    }
    bool operator!=(const ShogiMove& o) const { return !(*this == o); }

    // Needed for use as map key / sorting.
    bool operator<(const ShogiMove& o) const {
        return raw_.proFromAndTo() < o.raw_.proFromAndTo();
    }

private:
    Move raw_;
};

// ---------------------------------------------------------------------------
// ShogiPosition — thin wrapper over cshogi's Position.
//
// Wraps undo-state in a deque so callers do not need to manage StateInfo.
// ---------------------------------------------------------------------------
class ShogiPosition {
public:
    // Standard starting position SFEN.
    static constexpr const char* kStartSFEN =
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    // Construct from SFEN string (default = starting position).
    explicit ShogiPosition(const std::string& sfen = kStartSFEN);

    // Copy constructor — deep copies the cshogi Position and history.
    ShogiPosition(const ShogiPosition& other);
    ShogiPosition& operator=(const ShogiPosition& other);

    // Access the underlying cshogi Position (GPL-licensed callers only).
    const Position& raw() const { return pos_; }
    Position&       raw()       { return pos_; }

    // Apply / undo a move.  The StateInfo stack is maintained internally.
    void do_move  (ShogiMove m);
    void undo_move(ShogiMove m);


    // Side to move: true = Black (先手), false = White (後手).
    bool black_to_move() const { return pos_.turn() == Black; }

    // Zobrist hash of the full position (board + hand).
    std::uint64_t key() const { return pos_.getKey(); }

    // SFEN string.
    std::string to_sfen() const { return pos_.toSFEN(); }

    // Hand piece counts (for encoding).
    int hand_count(Color c, HandPiece hp) const {
        return static_cast<int>(pos_.hand(c).numOf(hp));
    }

private:
    Position pos_;
    // StateInfo history for undo support.
    std::deque<StateInfo> state_stack_;
};

// ---------------------------------------------------------------------------
// ShogiTraits — the Traits type plugged into GameRules<ShogiTraits>.
// ---------------------------------------------------------------------------
struct ShogiTraits {
    using Position = ShogiPosition;
    using Move     = ShogiMove;

    static constexpr const char* kEngineName = "shogi_engine";

    // Number of policy outputs (81 squares × 27 move types = 2187).
    static constexpr int kPolicySize = 2187;

    // NN tensor dimensions: [81 squares, 90 channels].
    static constexpr int kNumSquares  = 81;
    static constexpr int kNumChannels = 90;

    // --- GameRules interface ------------------------------------------------

    static void generate_legal(const ShogiPosition& pos,
                                core::MoveList<ShogiMove>& out);

    static void apply(ShogiPosition& pos, ShogiMove m);
    static void undo (ShogiPosition& pos, ShogiMove m);

    static bool  is_terminal   (const ShogiPosition& pos);
    static float terminal_value(const ShogiPosition& pos);

    static std::uint64_t hash(const ShogiPosition& pos);

    static void encode_nn(const ShogiPosition& pos, float* out);

    static int       move_to_policy_idx  (ShogiMove m, const ShogiPosition& pos);
    static ShogiMove policy_idx_to_move  (int idx,     const ShogiPosition& pos);

    // String representation (USI format, e.g. "7g7f", "P*5f").
    static std::string move_to_string(const ShogiMove& m) {
        return m.to_usi();
    }

    // Null-move check.
    static bool is_null_move(const ShogiMove& m) { return m.is_none(); }
};

// Convenience alias.
using ShogiRules = core::GameRules<ShogiTraits>;

} // namespace shogi
