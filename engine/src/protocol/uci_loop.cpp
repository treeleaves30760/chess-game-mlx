// SPDX-License-Identifier: MIT
// engine/src/protocol/uci_loop.cpp
//
// Legacy entry — the real templated UCI loop lives in uci_loop.hpp and is
// instantiated in chess_engine.  This translation unit provides a stubbed
// definition of run_uci_loop() so the chess_mlx_protocol library (which is
// game-agnostic) can link.  chess_engine's main() uses the header-only
// run_uci_loop_with_backend() helper directly.

namespace chess_mlx::protocol {

// Forward declaration — definition in header doesn't exist at link time
// from this TU; we just provide an empty legacy shim.
void run_uci_loop();

void run_uci_loop() {
    // Legacy stub — callers should use run_uci_loop_with_backend().
}

} // namespace chess_mlx::protocol
