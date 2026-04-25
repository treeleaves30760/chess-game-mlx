// SPDX-License-Identifier: MIT
// engine/include/nn/stub_backend.hpp
//
// StubBackend — an always-on fallback that returns a uniform policy, value 0,
// and a constant moves-left estimate.  Used when the MLX backend is disabled
// at compile time or when no weights are provided.  This lets MCTS development
// proceed without a functioning inference pipeline.

#pragma once

#include "nn/backend.hpp"

namespace chess_mlx::nn {

class StubBackend : public NNBackend {
public:
    // `policy_size`:   4672 (chess) or 2187 (shogi)
    // `input_size`:    64*19 (chess) or 81*90 (shogi) — cosmetic only
    StubBackend(std::size_t policy_size, std::size_t input_size);

    NNOutput              evaluate      (const std::vector<float>& input_tensor) override;
    std::vector<NNOutput> evaluate_batch(const std::vector<std::vector<float>>& inputs) override;

    std::string name()        const override { return "stub"; }
    std::size_t input_size()  const override { return input_size_; }
    std::size_t policy_size() const override { return policy_size_; }

private:
    std::size_t policy_size_;
    std::size_t input_size_;
};

} // namespace chess_mlx::nn
