// SPDX-License-Identifier: MIT
// engine/src/chess/main_chess.cpp
//
// Chess engine entry point.  Speaks UCI on stdin/stdout.
//
// CLI:
//   chess_engine [--weights /path/to/model.safetensors]
//                [--lc0-weights /path/to/BT4.onnx]
//                [--mcts-threads N] [--nn-threads N] [--threads N]
//                [--multipv K]
//
// Threads:
//   --mcts-threads N : MCTS worker thread count (PUCT descent + virtual loss).
//   --nn-threads N   : NN inference thread count.  Drives ORT
//                      SetIntraOpNumThreads() for the LC0 path; ignored by
//                      MLX backend (Metal dispatches on its own queue).
//   --threads N      : Legacy alias — sets BOTH mcts-threads and nn-threads
//                      to N if neither was given explicitly.
//
// Tuning tips for an M3 Air (8 cores):
//   * MLX backend          : --mcts-threads 4 --nn-threads 1
//                            (nn-threads is a no-op for MLX; what matters is
//                            keeping MCTS workers under the P-core count so
//                            they don't fight for cycles during PUCT descent.)
//   * LC0 ONNX-CPU backend : --mcts-threads 2 --nn-threads 6
//                            (each MCTS worker blocks on a future for ~600ms;
//                            give the remaining cores to ORT's MLAS matmul.)

#include "nn/backend.hpp"
#include "nn/backend_factory.hpp"
#include "protocol/chess_uci.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    std::string weights_path;
    std::string lc0_weights_path;
    int legacy_threads = 0;   // set when user passes --threads
    int mcts_threads = 0;     // 0 == "not explicitly set"
    int nn_threads = 0;
    int multipv = 1;

    // Simple CLI parser.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if ((arg == "--lc0-weights" || arg == "--lc0_weights") && i + 1 < argc) {
            lc0_weights_path = argv[++i];
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
                "Usage: chess_engine [--weights PATH] [--lc0-weights PATH]\n"
                "                    [--mcts-threads N] [--nn-threads N]\n"
                "                    [--threads N] [--multipv K]\n"
                "\n"
                "  --mcts-threads N  MCTS worker thread count.\n"
                "  --nn-threads N    NN inference intra-op threads (ORT only;\n"
                "                    ignored by MLX backend).\n"
                "  --threads N       Legacy alias: sets BOTH of the above to N\n"
                "                    when neither is given explicitly.\n";
            return 0;
        }
    }

    // Apply the legacy --threads alias to whichever knob wasn't set.
    if (legacy_threads > 0) {
        if (mcts_threads <= 0) mcts_threads = legacy_threads;
        if (nn_threads   <= 0) nn_threads   = legacy_threads;
    }
    if (mcts_threads <= 0) mcts_threads = 1;
    if (nn_threads   <= 0) nn_threads   = 1;

    chess_mlx::nn::BackendConfig bc;
    bc.game             = chess_mlx::nn::Game::Chess;
    bc.weights_path     = weights_path;
    bc.lc0_onnx_path    = lc0_weights_path;
    bc.nn_threads       = nn_threads;
    auto backend = std::shared_ptr<chess_mlx::nn::NNBackend>(
        chess_mlx::nn::make_nn_backend(bc).release());

    const std::size_t in_sz = backend->input_size();
    const std::size_t po_sz = backend->policy_size();

    std::cerr << "info string chess_engine started; backend=" << backend->name()
              << " input=" << in_sz << " policy=" << po_sz
              << " mcts_threads=" << mcts_threads
              << " nn_threads="   << nn_threads
              << " multipv=" << multipv << "\n";

    // -----------------------------------------------------------------------
    // Dispatch to the right protocol/traits combo based on the backend's
    // *self-reported* sizes (driven by the model sidecar for MLX, hard-coded
    // for LC0 ONNX).  See protocol/chess_uci.hpp for what each routine wires.
    //
    //   input=64*112 (=7168), policy=1858  → BT4-style LC0 ONNX  (stm-relative)
    //   input=64*19  (=1216), policy=1858  → MLX checkpoints     (compact 1858)
    //   input=64*19  (=1216), policy=4672  → native (no shipped model uses this)
    //   input=64*90  (shogi)               → handled by shogi_engine, not here
    // -----------------------------------------------------------------------
    constexpr std::size_t kLc0InputSize    = 64 * 112;  // 7168
    constexpr std::size_t kNativeInputSize = 64 *  19;  // 1216
    constexpr std::size_t kCompact1858     = 1858;
    constexpr std::size_t kNative4672      = 4672;

    if (in_sz == kLc0InputSize && po_sz == kCompact1858) {
        return chess_mlx::protocol::run_lc0_uci_loop_with_backend(
            backend, mcts_threads, multipv);
    }
    if (in_sz == kNativeInputSize && po_sz == kCompact1858) {
        return chess_mlx::protocol::run_hybrid_uci_loop_with_backend(
            backend, mcts_threads, multipv);
    }
    if (in_sz == kNativeInputSize && po_sz == kNative4672) {
        return chess_mlx::protocol::run_uci_loop_with_backend(
            backend, mcts_threads, multipv);
    }

    // Unrecognised shape — fall back to the native loop and log loudly so
    // the GUI surface notes it.  This keeps the engine alive (StubBackend
    // path) while giving the user a clear pointer to the misconfiguration.
    std::cerr << "info string chess_engine: unrecognised (input=" << in_sz
              << ", policy=" << po_sz
              << ") combo; defaulting to native traits — moves may be poor\n";
    return chess_mlx::protocol::run_uci_loop_with_backend(
        backend, mcts_threads, multipv);
}
