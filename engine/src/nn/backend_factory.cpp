// SPDX-License-Identifier: MIT
// engine/src/nn/backend_factory.cpp

#include "nn/backend_factory.hpp"
#include "nn/stub_backend.hpp"

#ifdef CHESS_MLX_HAS_MLX
#include "nn/mlx_backend.hpp"
#endif

#ifdef CHESS_MLX_HAS_ONNX
#include "nn/lc0_backend.hpp"
#endif

#include <iostream>

namespace chess_mlx::nn {

// Sizes per game:
//   Chess: input = 64 * 19 = 1216, engine policy = 4672
//   Shogi: input = 81 * 90 = 7290, engine policy = 2187
// LC0 backend always uses chess: input = 64 * 112 = 7168, policy = 4672
static constexpr std::size_t kChessInputSize     = 64 * 19;
static constexpr std::size_t kChessPolicySize    = 4672;
static constexpr std::size_t kShogiInputSize     = 81 * 90;
static constexpr std::size_t kShogiPolicySize    = 2187;
static constexpr std::size_t kLc0ChessInputSize  = 64 * 112;  // 7168

bool mlx_available() noexcept {
#ifdef CHESS_MLX_HAS_MLX
    return true;
#else
    return false;
#endif
}

bool onnx_available() noexcept {
#ifdef CHESS_MLX_HAS_ONNX
    return true;
#else
    return false;
#endif
}

std::unique_ptr<NNBackend> make_nn_backend(const BackendConfig& cfg) {
    // Resolve effective NN-thread count: prefer the explicit `nn_threads`;
    // honour the deprecated `lc0_threads` only if it was set non-zero AND
    // `nn_threads` was left at its default of 1.
    const int nn_threads_eff =
        (cfg.nn_threads > 1 || cfg.lc0_threads <= 0) ? cfg.nn_threads
                                                     : cfg.lc0_threads;

    // LC0 path takes priority when specified.
    if (!cfg.lc0_onnx_path.empty() && !cfg.force_stub) {
#ifdef CHESS_MLX_HAS_ONNX
        try {
            return std::make_unique<Lc0Backend>(
                cfg.lc0_onnx_path, nn_threads_eff, /*inter_threads=*/1);
        } catch (const std::exception& e) {
            std::cerr << "info string LC0/ONNX backend failed: "
                      << e.what() << "; falling back to StubBackend\n";
            return std::make_unique<StubBackend>(kChessPolicySize, kLc0ChessInputSize);
        }
#else
        std::cerr << "info string LC0/ONNX backend not compiled in "
                  << "(CHESS_MLX_WITH_ONNX=OFF); using StubBackend\n";
        return std::make_unique<StubBackend>(kChessPolicySize, kLc0ChessInputSize);
#endif
    }

    const auto [input_size, policy_size] =
        (cfg.game == Game::Chess)
            ? std::pair<std::size_t, std::size_t>{kChessInputSize, kChessPolicySize}
            : std::pair<std::size_t, std::size_t>{kShogiInputSize, kShogiPolicySize};

    if (cfg.force_stub || cfg.weights_path.empty()) {
        return std::make_unique<StubBackend>(policy_size, input_size);
    }

#ifdef CHESS_MLX_HAS_MLX
    try {
        return std::make_unique<MlxBackend>(cfg.weights_path, policy_size, input_size);
    } catch (const std::exception& e) {
        std::cerr << "info string MLX backend failed to load weights: "
                  << e.what() << "; falling back to StubBackend\n";
        return std::make_unique<StubBackend>(policy_size, input_size);
    }
#else
    std::cerr << "info string MLX backend not compiled in "
              << "(CHESS_MLX_WITH_MLX=OFF); using StubBackend\n";
    return std::make_unique<StubBackend>(policy_size, input_size);
#endif
}

} // namespace chess_mlx::nn
