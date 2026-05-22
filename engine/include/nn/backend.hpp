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
// Logit access goes through `policy_at(i)`.  Two storage modes are supported:
//
//   1. Owned (`policy` non-empty): the result holds its own logit buffer.
//      Used by `evaluate()` and any backend that doesn't share a batch
//      buffer.  Simple but allocates per-leaf.
//
//   2. Shared (`policy_shared` set): a batch of leaves all reference one
//      contiguous logit tensor, each leaf reading at its own
//      `[policy_offset, policy_offset+policy_extent)` slice.  This eliminates
//      the per-leaf 1858/4672-float malloc that used to dominate the
//      post-GPU CPU portion of the search loop.
//
// `value`       is in [-1, +1], positive = white / 先手 advantage.
// `moves_left`  is a non-negative scalar (half-moves remaining estimate).
struct NNOutput {
    // Owned mode storage.  Empty when shared mode is active.
    std::vector<float> policy;

    // Shared mode: one contiguous buffer for an entire batch; each leaf
    // reads its own slice.  When non-null this overrides `policy`.
    std::shared_ptr<const std::vector<float>> policy_shared{};
    std::size_t policy_offset{0};   // start index inside *policy_shared
    std::size_t policy_extent{0};   // number of logits this leaf owns

    float              value{0.0f};
    float              moves_left{50.0f};

    // Effective number of logits available to the caller.
    [[nodiscard]] std::size_t policy_size() const noexcept {
        return policy_shared ? policy_extent : policy.size();
    }

    // Read a single logit by index.  Out-of-range reads return 0.0f, matching
    // the legacy "absent slot" convention so callers don't have to bounds-check.
    [[nodiscard]] float policy_at(std::size_t i) const noexcept {
        if (policy_shared) {
            return (i < policy_extent)
                ? (*policy_shared)[policy_offset + i]
                : 0.0f;
        }
        return (i < policy.size()) ? policy[i] : 0.0f;
    }
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
