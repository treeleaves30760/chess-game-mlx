// SPDX-License-Identifier: MIT
// engine/src/nn/backend_stub.cpp
//
// StubBackend implementation — uniform policy, value 0, moves_left 50.
// Used when no NN is available.  Keeps the MCTS pipeline functional.

#include "nn/stub_backend.hpp"

namespace chess_mlx::nn {

StubBackend::StubBackend(std::size_t policy_size, std::size_t input_size)
    : policy_size_(policy_size), input_size_(input_size) {}

NNOutput StubBackend::evaluate(const std::vector<float>& /*input_tensor*/) {
    NNOutput out;
    out.policy.assign(policy_size_, 0.0f);
    out.value      = 0.0f;
    out.moves_left = 50.0f;
    return out;
}

std::vector<NNOutput> StubBackend::evaluate_batch(
    const std::vector<std::vector<float>>& inputs) {
    std::vector<NNOutput> results;
    results.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        results.push_back(evaluate(inputs[i]));
    }
    return results;
}

// Keep the original anchor symbol for linker compatibility.
void backend_stub_anchor() noexcept {}

} // namespace chess_mlx::nn
