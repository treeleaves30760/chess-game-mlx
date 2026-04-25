// engine/src/search/multi_ponder.cpp
//
// MultiPonderManager<Traits> is a template class defined entirely in
// multi_ponder.hpp.  This translation unit exists so that the CMake target
// chess_mlx_search always has at least one .cpp file to compile.

namespace chess_mlx::search {
// Linker anchor — keeps the library target non-empty.
void multi_ponder_anchor() noexcept {}
} // namespace chess_mlx::search
