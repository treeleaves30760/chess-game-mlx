// GPL-3.0-or-later wrapper for cshogi upstream hand.cpp
// Compiles the upstream source with Apple Clang diagnostic suppressions.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#pragma clang diagnostic ignored "-Wno-invalid-constexpr"
#endif
#include "../cshogi-upstream/src/hand.cpp"
#ifdef __clang__
#pragma clang diagnostic pop
#endif
