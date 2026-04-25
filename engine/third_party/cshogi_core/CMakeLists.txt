# Minimal CMake shim that compiles just the move-gen + position core from cshogi-upstream.
# This library is GPL v3 (derived from Apery/Stockfish). Only link it into shogi_engine.

set(CSHOGI_UPSTREAM ${CMAKE_CURRENT_LIST_DIR}/../cshogi-upstream/src)

if(NOT EXISTS ${CSHOGI_UPSTREAM}/bitboard.cpp)
    message(FATAL_ERROR "cshogi upstream sources not found at ${CSHOGI_UPSTREAM}. "
                        "Did you run `git submodule update --init`?")
endif()

# Only the files needed for Position + move generation + hand pieces + USI parsing.
# Excluded: book.cpp, dfpn.cpp, mate.cpp, search.cpp (engine search we replace).
# usi.cpp is included: it provides usiToMove() used by our shogi/usi.hpp wrappers.
# DefaultStartPositionSFEN is defined in usi.hpp (included only by usi.cpp), so
# there is exactly one definition — no ODR violation.
add_library(cshogi_core STATIC
    ${CSHOGI_UPSTREAM}/bitboard.cpp
    ${CSHOGI_UPSTREAM}/common.cpp
    ${CSHOGI_UPSTREAM}/generateMoves.cpp
    ${CSHOGI_UPSTREAM}/hand.cpp
    ${CSHOGI_UPSTREAM}/init.cpp
    ${CSHOGI_UPSTREAM}/move.cpp
    ${CSHOGI_UPSTREAM}/mt64bit.cpp
    ${CSHOGI_UPSTREAM}/position.cpp
    ${CSHOGI_UPSTREAM}/square.cpp
    ${CSHOGI_UPSTREAM}/usi.cpp
)

target_include_directories(cshogi_core PUBLIC ${CSHOGI_UPSTREAM})

# Apple Silicon: don't enable x86 SSE/AVX defines. The cshogi bitboard code has a
# scalar fallback path that works on arm64 (guarded with __aarch64__).
#
# cshogi was originally written for C++11.  Its SquareWithWall enum initialises
# values using SquareDelta arithmetic that produces results outside the declared
# enum range.  In C++17, clang's constexpr evaluator treats this as a hard error
# ("not a constant expression").  We compile the cshogi TUs with -std=c++14
# (PRIVATE, so our C++17 code is unaffected) and -w to silence warnings.
# The PUBLIC include dir does NOT force C++14 on consumers — they still compile
# at C++17 and simply include the cshogi headers, which only triggers the issue
# at definition-time (in the TU that declares the SquareWithWall enum), not at
# use-time.
# Consumers of cshogi_core must compile at C++17 or later.
# The cshogi TUs themselves build at whatever standard the project sets.
target_compile_features(cshogi_core PUBLIC cxx_std_17)

# Silence all diagnostics from vendored code.
# (square.hpp has been patched to avoid the CWG2518 SquareDelta constexpr
# out-of-range issue that Apple Clang 21+ enforces — see the comment in
# the patched square.hpp line 290.)
target_compile_options(cshogi_core PRIVATE
    -w
)

# On arm64 (Apple Silicon) we must NOT pass -mavx2 / -msse4.2.
# cshogi's setup.py hardcodes these; we simply don't add them here.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    target_compile_definitions(cshogi_core PRIVATE
        # intentionally leave HAVE_SSE*, HAVE_AVX*, HAVE_BMI2 undefined
    )
endif()

# Make cshogi's GPL taint visible to callers.
target_compile_definitions(cshogi_core PUBLIC CHESS_MLX_SHOGI_IS_GPL=1)
