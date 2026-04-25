// SPDX-License-Identifier: GPL-3.0-or-later
// engine/include/protocol/usi_loop.hpp
//
// Legacy USI loop declaration.  Real USI loop helpers are in
// protocol/shogi_usi.hpp (guarded by GPL v3); the templated driver lives in
// protocol/uci_loop.hpp.

#pragma once

namespace chess_mlx::protocol {

// Legacy stub entry point (unused by new code).
void run_usi_loop();

} // namespace chess_mlx::protocol
