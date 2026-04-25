// SPDX-License-Identifier: MIT
// engine/include/search/policy_utils.hpp
//
// Lightweight policy helpers for multi-ponder root prediction.
//
// top_k_policy_moves<Traits>: given a position, do one NN forward pass and
// return the top-k legal moves sorted by softmax probability.  This is
// "Option 3" from the Phase 6 spec: cheapest approach that is still
// NN-quality for v1.

#pragma once

#include "core/game_rules.hpp"
#include "nn/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace chess_mlx::search {

// ---------------------------------------------------------------------------
// top_k_policy_moves
//
// Returns up to `k` (move, probability) pairs sorted descending by
// softmax probability under the NN policy head.
//
// Template parameters:
//   Traits  — ChessTraits or ShogiTraits; must provide:
//               using Position, using Move
//               static void generate_legal(Position&, MoveList&)
//               static void encode_nn(Position&, float*)
//               static int  move_to_policy_idx(Move, Position&)
//
// Arguments:
//   backend    — NN backend (evaluate_batch preferred when batch_preferred()).
//   pos        — position to evaluate (opponent's turn).
//   input_size — flat input tensor size (e.g. 64*19 = 1216 for chess).
//   k          — how many top moves to return (clamped to legal-move count).
// ---------------------------------------------------------------------------
template <typename Traits>
std::vector<std::pair<typename Traits::Move, float>>
top_k_policy_moves(nn::NNBackend& backend,
                   const typename Traits::Position& pos,
                   std::size_t input_size,
                   int k) {
    using Move     = typename Traits::Move;
    using MoveList = core::MoveList<Move>;

    // 1. Generate legal moves.
    MoveList ml;
    Traits::generate_legal(pos, ml);
    if (ml.empty()) return {};

    // 2. Encode position.
    std::vector<float> input(input_size, 0.0f);
    Traits::encode_nn(pos, input.data());

    // 3. NN forward (single-position).
    nn::NNOutput out;
    if (backend.batch_preferred()) {
        out = backend.evaluate_batch({input}).at(0);
    } else {
        out = backend.evaluate(input);
    }

    // 4. Gather logits for legal moves and apply softmax.
    const std::size_t policy_sz = out.policy.size();
    std::vector<float> logits;
    logits.reserve(ml.size());
    float max_l = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < ml.size(); ++i) {
        const int idx = Traits::move_to_policy_idx(ml[i], pos);
        const float l = (idx >= 0 && static_cast<std::size_t>(idx) < policy_sz)
                       ? out.policy[static_cast<std::size_t>(idx)]
                       : 0.0f;
        logits.push_back(l);
        if (l > max_l) max_l = l;
    }
    double denom = 0.0;
    for (auto& l : logits) { l = std::exp(l - max_l); denom += l; }
    if (denom > 0.0) {
        for (auto& l : logits) l = static_cast<float>(static_cast<double>(l) / denom);
    } else {
        const float uni = 1.0f / static_cast<float>(ml.size());
        for (auto& l : logits) l = uni;
    }

    // 5. Collect and sort.
    std::vector<std::pair<Move, float>> pairs;
    pairs.reserve(ml.size());
    for (std::size_t i = 0; i < ml.size(); ++i) {
        pairs.emplace_back(ml[i], logits[i]);
    }
    const std::size_t top = static_cast<std::size_t>(
        std::min(k, static_cast<int>(pairs.size())));
    std::partial_sort(pairs.begin(),
                      pairs.begin() + static_cast<std::ptrdiff_t>(top),
                      pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.second > b.second;
                      });
    pairs.resize(top);
    return pairs;
}

} // namespace chess_mlx::search
