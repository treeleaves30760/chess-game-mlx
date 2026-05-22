// SPDX-License-Identifier: MIT
// engine/include/nn/backend_factory.hpp
//
// Factory helper that picks an NNBackend based on the game, desired weights
// file, and whether MLX-C was built in.

#pragma once

#include "nn/backend.hpp"

#include <memory>
#include <string>

namespace chess_mlx::nn {

enum class Game { Chess, Shogi };

struct BackendConfig {
    Game        game{Game::Chess};
    std::string weights_path;    // MLX safetensors path — empty => StubBackend
    std::string lc0_onnx_path;  // LC0 ONNX path — takes priority over weights_path
    bool        force_stub{false};
    // Threads hint for the NN backend's *inference* parallelism, e.g. ORT's
    // SetIntraOpNumThreads for Lc0Backend.  Distinct from MCTS worker thread
    // count, which is configured separately on the MCTS instance.  For MLX
    // backend this is ignored (Metal dispatches on its own queue).
    int         nn_threads{1};

    // Deprecated alias retained for source compatibility.  When non-zero
    // overrides `nn_threads` if `nn_threads` is left at its default of 1.
    int         lc0_threads{0};
};

// Create the best-available backend given the config.
//   * lc0_onnx_path non-empty  → Lc0Backend  (ONNX Runtime, if compiled in)
//   * weights_path non-empty   → MlxBackend   (MLX-C, if compiled in)
//   * otherwise                → StubBackend  (uniform policy)
// The helper emits `info string ...` diagnostics on stderr on fallback.
std::unique_ptr<NNBackend> make_nn_backend(const BackendConfig& cfg);

// Query whether MLX-C was compiled in.
bool mlx_available() noexcept;

// Query whether ONNX Runtime was compiled in.
bool onnx_available() noexcept;

} // namespace chess_mlx::nn
