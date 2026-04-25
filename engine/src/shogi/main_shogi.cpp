// SPDX-License-Identifier: GPL-3.0-or-later
// engine/src/shogi/main_shogi.cpp
//
// Shogi engine entry point.  Speaks USI on stdin/stdout.
//
// LICENSE NOTE: Links against cshogi (GPL v3) — the shogi_engine binary is
// therefore GPL v3.  Do NOT link this translation unit into chess_engine.

#include "nn/backend.hpp"
#include "nn/backend_factory.hpp"
#include "nn/stub_backend.hpp"
#include "protocol/shogi_usi.hpp"
#include "shogi/shogi_traits.hpp"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    // Ensure cshogi global tables are initialised before anything else.
    ::shogi::ShogiRuntime::ensure_initialized();

    std::string weights_path;
    int threads = 1;
    int multipv = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            try { threads = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--multipv" && i + 1 < argc) {
            try { multipv = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: shogi_engine [--weights PATH] [--threads N] [--multipv K]\n";
            return 0;
        }
    }

    chess_mlx::nn::BackendConfig bc;
    bc.game         = chess_mlx::nn::Game::Shogi;
    bc.weights_path = weights_path;
    auto backend = std::shared_ptr<chess_mlx::nn::NNBackend>(
        chess_mlx::nn::make_nn_backend(bc).release());

    std::cerr << "info string shogi_engine started; backend=" << backend->name()
              << " threads=" << threads << " multipv=" << multipv << "\n";

    return chess_mlx::protocol::run_usi_loop_with_backend(backend, threads, multipv);
}
