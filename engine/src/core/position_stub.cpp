// engine/src/core/position_stub.cpp
//
// Intentionally empty translation unit — exists only to satisfy the CMake
// chess_mlx_core STATIC target, which requires at least one .cpp file so
// that ar does not produce an empty archive (a warning on some linkers).
//
// A trivial symbol is provided to silence the "empty translation unit" warning
// that some compilers (clang with -pedantic) emit.

namespace chess_mlx {
namespace detail {

// A single dummy linkage symbol; never called.
void position_stub_anchor() noexcept {}

} // namespace detail
} // namespace chess_mlx
