// SPDX-License-Identifier: GPL-3.0-or-later
// engine/src/shogi/main_shogi.cpp
//
// Shogi engine entry point.  Speaks USI on stdin/stdout.
//
// LICENSE NOTE: Links against cshogi (GPL v3) — the shogi_engine binary is
// therefore GPL v3.  Do NOT link this translation unit into chess_engine.
//
// CLI mirrors main_chess.cpp.  See that file for the rationale behind
// --mcts-threads vs --nn-threads.

#include "nn/backend.hpp"
#include "nn/backend_factory.hpp"
#include "protocol/shogi_usi.hpp"
#include "shogi/shogi_traits.hpp"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    // Ensure cshogi global tables are initialised before anything else.
    ::shogi::ShogiRuntime::ensure_initialized();

    std::string weights_path;
    int legacy_threads = 0;
    int mcts_threads = 0;
    int nn_threads = 0;
    int multipv = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if ((arg == "--mcts-threads" || arg == "--mcts_threads") && i + 1 < argc) {
            try { mcts_threads = std::stoi(argv[++i]); } catch (...) {}
        } else if ((arg == "--nn-threads" || arg == "--nn_threads") && i + 1 < argc) {
            try { nn_threads = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--threads" && i + 1 < argc) {
            try { legacy_threads = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--multipv" && i + 1 < argc) {
            try { multipv = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: shogi_engine [--weights PATH]\n"
                "                    [--mcts-threads N] [--nn-threads N]\n"
                "                    [--threads N] [--multipv K]\n";
            return 0;
        }
    }

    if (legacy_threads > 0) {
        if (mcts_threads <= 0) mcts_threads = legacy_threads;
        if (nn_threads   <= 0) nn_threads   = legacy_threads;
    }
    if (mcts_threads <= 0) mcts_threads = 1;
    if (nn_threads   <= 0) nn_threads   = 1;

    chess_mlx::nn::BackendConfig bc;
    bc.game         = chess_mlx::nn::Game::Shogi;
    bc.weights_path = weights_path;
    bc.nn_threads   = nn_threads;
    auto backend = std::shared_ptr<chess_mlx::nn::NNBackend>(
        chess_mlx::nn::make_nn_backend(bc).release());

    std::cerr << "info string shogi_engine started; backend=" << backend->name()
              << " mcts_threads=" << mcts_threads
              << " nn_threads="   << nn_threads
              << " multipv=" << multipv << "\n";

    return chess_mlx::protocol::run_usi_loop_with_backend(backend, mcts_threads, multipv);
}
