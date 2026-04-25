// SPDX-License-Identifier: MIT
// engine/include/nn/backend.hpp
//
// Abstract NNBackend interface.  Implementations:
//   - StubBackend  (always available, returns uniform policy)
//   - MlxBackend   (MLX-C based, loads a .safetensors checkpoint)
//
// The backend is game-agnostic; the caller is responsible for feeding in the
// right-sized input tensor and interpreting the output policy array.  The
// `policy_size` / `input_size` knob is passed to the backend at construction.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace chess_mlx::nn {

// ---------------------------------------------------------------------------
// NNOutput — a single evaluation result.
// ---------------------------------------------------------------------------
//
// `policy` is a dense logit-style array of size `policy_size`.  The caller
// converts it to probabilities (softmax + legal-move mask).
// `value`       is in [-1, +1], positive = white / 先手 advantage.
// `moves_left`  is a non-negative scalar (half-moves remaining estimate).
struct NNOutput {
    std::vector<float> policy;   // size = 4672 (chess) or 2187 (shogi)
    float              value{0.0f};
    float              moves_left{50.0f};
};

// ---------------------------------------------------------------------------
// NNBackend — abstract interface for neural-network inference.
// ---------------------------------------------------------------------------
class NNBackend {
public:
    virtual ~NNBackend() = default;

    // Evaluate a single input.  `input_tensor` is a flat float32 array of
    // size `input_size()` (64*19 = 1216 for chess, 81*90 = 7290 for shogi).
    virtual NNOutput evaluate(const std::vector<float>& input_tensor) = 0;

    // Evaluate a batch.  May be more efficient than N individual calls.
    virtual std::vector<NNOutput> evaluate_batch(
        const std::vector<std::vector<float>>& inputs) = 0;

    // Short human-readable name ("stub", "mlx-chess", ...).
    virtual std::string name() const = 0;

    // Sizes declared by the backend (used by callers to pre-allocate buffers).
    virtual std::size_t input_size()  const = 0;
    virtual std::size_t policy_size() const = 0;

    // Whether this backend benefits from batching.  Set to false for the
    // stub (trivial work) so the MCTS layer can bypass the batcher and
    // avoid thread-context-switch overhead.  MLX backends should return
    // true — running a single forward is expensive, so batching helps.
    virtual bool batch_preferred() const { return false; }
};

} // namespace chess_mlx::nn
