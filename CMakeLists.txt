cmake_minimum_required(VERSION 3.25)
project(chess_game_mlx
    VERSION 0.1.0
    DESCRIPTION "Chess + Shogi AI engine with MLX-C inference"
    LANGUAGES CXX
)

# ------------------------------------------------------------------
# Global build settings
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# macOS-specific: universal binary not needed (Apple Silicon only)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")
endif()

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(CHESS_MLX_BUILD_TESTS "Build unit tests" ON)
option(CHESS_MLX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CHESS_MLX_WITH_MLX "Enable MLX-C NN backend (requires mlx-c submodule)" ON)
option(CHESS_MLX_WITH_ONNX "Enable ONNX Runtime LC0 backend" ON)

# ------------------------------------------------------------------
# ONNX Runtime detection (optional, for LC0 backend).
# Must happen BEFORE add_subdirectory(engine) so the engine CMakeLists
# can see OnnxRuntime_FOUND.
# ------------------------------------------------------------------
if(CHESS_MLX_WITH_ONNX)
    # Always use explicit find_path / find_library to get full paths.
    # This avoids the pkg-config "-lonnxruntime" without -L issue.
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS /opt/homebrew/include/onnxruntime
              /opt/homebrew/Cellar/onnxruntime/1.25.0/include/onnxruntime
              /usr/local/include/onnxruntime
              ${CMAKE_PREFIX_PATH}/include/onnxruntime
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime libonnxruntime
        HINTS /opt/homebrew/lib
              /opt/homebrew/Cellar/onnxruntime/1.25.0/lib
              /usr/local/lib
              ${CMAKE_PREFIX_PATH}/lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        set(OnnxRuntime_FOUND TRUE)
        set(OnnxRuntime_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
        set(OnnxRuntime_LIBRARIES    ${ONNXRUNTIME_LIB})
        get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB} DIRECTORY)
        message(STATUS "ONNX Runtime found — LC0 backend enabled")
        message(STATUS "  ONNX include: ${OnnxRuntime_INCLUDE_DIRS}")
        message(STATUS "  ONNX lib:     ${OnnxRuntime_LIB}")
        message(STATUS "  ONNX lib dir: ${ONNXRUNTIME_LIB_DIR}")
    else()
        message(WARNING "ONNX Runtime not found — LC0 backend disabled. "
                        "Install via 'brew install onnxruntime' or set "
                        "CMAKE_PREFIX_PATH=/opt/homebrew")
        set(CHESS_MLX_WITH_ONNX OFF)
    endif()
endif()

# ------------------------------------------------------------------
# Warnings
# ------------------------------------------------------------------
add_library(chess_mlx_warnings INTERFACE)
target_compile_options(chess_mlx_warnings INTERFACE
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion -Wsign-conversion
    -Wno-unused-parameter
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

# ------------------------------------------------------------------
# Testing — enable_testing() must be called BEFORE add_subdirectory()
# so that CTest generates CTestTestfile.cmake in the right directories.
# ------------------------------------------------------------------
if(CHESS_MLX_BUILD_TESTS)
    enable_testing()
endif()

# ------------------------------------------------------------------
# Sub-directories
# ------------------------------------------------------------------
add_subdirectory(engine)

# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------
message(STATUS "============================================================")
message(STATUS "Chess_Game_mlx configuration")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests           : ${CHESS_MLX_BUILD_TESTS}")
message(STATUS "  Benchmarks      : ${CHESS_MLX_BUILD_BENCHMARKS}")
message(STATUS "  MLX backend     : ${CHESS_MLX_WITH_MLX}")
message(STATUS "  ONNX backend    : ${CHESS_MLX_WITH_ONNX}")
message(STATUS "============================================================")
