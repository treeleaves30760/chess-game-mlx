// SPDX-License-Identifier: GPL-3.0-or-later
// engine/include/shogi/usi.hpp
//
// USI / SFEN helpers for the shogi engine.
//
// LICENSE NOTE: This file includes cshogi headers (GPL v3).
// Only link into shogi_engine, never chess_engine.
//
// Thin wrappers around cshogi's usi.cpp / position.cpp APIs.
// cshogi already has full SFEN parsing and USI move serialisation —
// we just surface them through our ShogiPosition / ShogiMove types.
//
// NOTE: We do NOT include cshogi's "usi.hpp" here — it defines
// DefaultStartPositionSFEN as a non-inline const std::string (ODR violation
// if included in multiple TUs) and its filename collides with ours.
// Instead we forward-declare usiToMove and use ShogiPosition::kStartSFEN.

#pragma once

#include "shogi/shogi_traits.hpp"

// cshogi headers already pulled in via shogi_traits.hpp

// Forward-declare cshogi's usiToMove (defined in cshogi-upstream/src/usi.cpp).
// We cannot include cshogi's usi.hpp because its filename "usi.hpp" collides
// with this file on case-insensitive (macOS) filesystems.
// Position is already fully defined via shogi_traits.hpp → position.hpp.
::Move usiToMove(const Position& pos, const std::string& moveStr);

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

namespace shogi {
namespace usi {

// ---------------------------------------------------------------------------
// SFEN parsing
// ---------------------------------------------------------------------------

// Parse a raw SFEN string and return a ShogiPosition.
// Throws std::runtime_error on invalid SFEN.
inline ShogiPosition position_from_sfen(const std::string& sfen) {
    ShogiRuntime::ensure_initialized();
    return ShogiPosition(sfen);
}

// Parse the "startpos" or "sfen <...>" part of a USI position command,
// optionally followed by "moves m1 m2 ...".
// Returns a ShogiPosition with all moves applied.
// Throws std::runtime_error on parse failure.
inline ShogiPosition position_from_usi_command(const std::string& cmd) {
    ShogiRuntime::ensure_initialized();
    std::istringstream ss(cmd);
    std::string token;
    std::string sfen;

    ss >> token; // "position" keyword — optional, skip if present
    if (token == "position") ss >> token;

    if (token == "startpos") {
        sfen = ShogiPosition::kStartSFEN;
        if (!(ss >> token)) return ShogiPosition(sfen); // no moves
        // token should be "moves"
    } else if (token == "sfen") {
        while (ss >> token && token != "moves")
            sfen += token + " ";
        // trim trailing space
        if (!sfen.empty() && sfen.back() == ' ')
            sfen.pop_back();
        if (token != "moves") return ShogiPosition(sfen); // no moves
    } else {
        throw std::runtime_error("usi: expected 'startpos' or 'sfen', got '" + token + "'");
    }

    ShogiPosition pos(sfen);

    // Apply moves.
    while (ss >> token) {
        const ::Move raw = usiToMove(pos.raw(), token);
        if (raw.value() == ::Move::MoveNone)
            throw std::runtime_error("usi: invalid move '" + token + "'");
        pos.do_move(ShogiMove(raw));
    }

    return pos;
}

// ---------------------------------------------------------------------------
// USI move serialisation
// ---------------------------------------------------------------------------

// Convert a ShogiMove to its USI string (e.g. "7g7f", "P*5f", "2b3c+").
inline std::string move_to_usi(ShogiMove m) {
    return m.to_usi();
}

// Parse a USI move string for the given position.
// Returns ShogiMove(MoveNone) if the string is not valid.
inline ShogiMove move_from_usi(const ShogiPosition& pos,
                                const std::string& usi_str) {
    const ::Move raw = usiToMove(pos.raw(), usi_str);
    return ShogiMove(raw);
}

// ---------------------------------------------------------------------------
// Legal-move helpers
// ---------------------------------------------------------------------------

// Returns all legal moves from `pos` as a vector of USI strings.
inline std::vector<std::string> legal_moves_usi(const ShogiPosition& pos) {
    const MoveList<LegalAll> ml(pos.raw());
    std::vector<std::string> result;
    result.reserve(ml.size());
    for (const ExtMove* it = const_cast<MoveList<LegalAll>&>(ml).begin();
         it != const_cast<MoveList<LegalAll>&>(ml).begin() + ml.size(); ++it) {
        result.push_back(it->move.toUSI());
    }
    return result;
}

// ---------------------------------------------------------------------------
// Default starting position SFEN
// ---------------------------------------------------------------------------
inline const std::string& start_sfen() {
    static const std::string s(ShogiPosition::kStartSFEN);
    return s;
}

} // namespace usi
} // namespace shogi
