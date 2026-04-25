// SPDX-License-Identifier: MIT
// engine/include/nn/mlx_backend.hpp
//
// MlxBackend — loads a ChessShogiTransformer checkpoint (.safetensors) via
// the MLX-C API and runs forward passes on the Apple GPU.
//
// If CHESS_MLX_HAS_MLX is not defined, this class is still declared but its
// constructor always throws std::runtime_error; callers should fall back to
// StubBackend.

#pragma once

#include "nn/backend.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace chess_mlx::nn {

// Architecture configuration parsed from the JSON sidecar of a checkpoint.
struct MlxModelConfig {
    std::string game;       // "chess" | "shogi"
    int         n_layers{12};
    int         d_model{512};
    int         n_heads{8};
    int         ffn_dim{2048};
    int         seq_len{64};
    int         feat_dim{19};
    int         num_moves{1858};
};

class MlxBackend : public NNBackend {
public:
    // Load weights from `weights_path` (must end in .safetensors).
    // Reads the sidecar `weights_path.with_suffix(".json")` for config.
    //
    // `engine_policy_size` is the policy size the *engine* uses (4672 for
    // chess, 2187 for shogi).  If the model's num_moves differs, the backend
    // projects the model's logits into the engine's wider slot space by
    // zero-filling — this is a stopgap until the compact LC0 1858-mapping is
    // wired through.  For shogi (2187 == 2187) they should match.
    //
    // Throws std::runtime_error on any load failure.
    MlxBackend(const std::string& weights_path,
               std::size_t        engine_policy_size,
               std::size_t        input_size);
    ~MlxBackend() override;

    NNOutput              evaluate      (const std::vector<float>& input_tensor) override;
    std::vector<NNOutput> evaluate_batch(const std::vector<std::vector<float>>& inputs) override;

    std::string name()        const override;
    std::size_t input_size()  const override { return input_size_; }
    std::size_t policy_size() const override { return engine_policy_size_; }
    bool        batch_preferred() const override { return true; }

    // Access loaded config (for diagnostics).
    const MlxModelConfig& config() const { return config_; }

private:
    // Implementation details are hidden behind an opaque pointer so that
    // consumers don't need to include mlx/c/mlx.h transitively.
    struct Impl;
    std::unique_ptr<Impl> impl_;

    MlxModelConfig config_;
    std::size_t    engine_policy_size_{0};
    std::size_t    input_size_{0};
    std::string    name_;

    // Serialise all calls into the MLX runtime — while MLX itself is
    // async/threadsafe for array ops, the safe bet for our first cut is
    // "one evaluation at a time".  The Batcher accumulates concurrent leaf
    // requests into a single batched call, so this is not a throughput
    // bottleneck in practice.
    mutable std::mutex mlx_mutex_;

    // Internal batched forward pass — caller MUST hold mlx_mutex_.
    // Processes B inputs at once; `flat` is all inputs concatenated (size B*S*F).
    // Returns B NNOutput values.
    std::vector<NNOutput> forward_batch_nolock(int B,
                                               const std::vector<float>& flat) const;
};

} // namespace chess_mlx::nn
