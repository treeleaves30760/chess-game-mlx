// SPDX-License-Identifier: MIT
// engine/src/chess/main_chess.cpp
//
// Chess engine entry point.  Speaks UCI on stdin/stdout.
//
// CLI:
//   chess_engine [--weights /path/to/model.safetensors] [--threads N]
//   chess_engine [--lc0-weights /path/to/BT4.onnx]     [--threads N]

#include "nn/backend.hpp"
#include "nn/backend_factory.hpp"
#include "nn/stub_backend.hpp"
#include "protocol/chess_uci.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    std::string weights_path;
    std::string lc0_weights_path;
    int threads = 1;
    int multipv = 1;

    // Simple CLI parser.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if ((arg == "--lc0-weights" || arg == "--lc0_weights") && i + 1 < argc) {
            lc0_weights_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            try { threads = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--multipv" && i + 1 < argc) {
            try { multipv = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: chess_engine [--weights PATH] [--lc0-weights PATH] "
                         "[--threads N] [--multipv K]\n";
            return 0;
        }
    }

    chess_mlx::nn::BackendConfig bc;
    bc.game             = chess_mlx::nn::Game::Chess;
    bc.weights_path     = weights_path;
    bc.lc0_onnx_path    = lc0_weights_path;
    bc.lc0_threads      = threads;
    auto backend = std::shared_ptr<chess_mlx::nn::NNBackend>(
        chess_mlx::nn::make_nn_backend(bc).release());

    std::cerr << "info string chess_engine started; backend=" << backend->name()
              << " threads=" << threads << " multipv=" << multipv << "\n";

    // Choose the correct MCTS traits based on which backend is active.
    if (!lc0_weights_path.empty() && backend->name() == "lc0-onnx") {
        // LC0 backend: 112-plane encoder, 1858-slot policy space.
        return chess_mlx::protocol::run_lc0_uci_loop_with_backend(
            backend, threads, multipv);
    }

    // Native (19-plane / 4672-slot) backend.
    return chess_mlx::protocol::run_uci_loop_with_backend(
        backend, threads, multipv);
}
