// SPDX-License-Identifier: GPL-3.0-or-later
// engine/src/protocol/usi_loop.cpp
//
// Legacy entry — the real templated USI loop lives in usi_loop.hpp and is
// instantiated in shogi_engine.

namespace chess_mlx::protocol {

void run_usi_loop();

void run_usi_loop() {
    // Legacy stub — callers should use run_usi_loop_with_backend().
}

} // namespace chess_mlx::protocol
