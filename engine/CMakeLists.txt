# ------------------------------------------------------------------
# Engine library target
# ------------------------------------------------------------------

# --------------- Third-party move-gen libraries ---------------
# chess-library (Disservin) is header-only.
set(CHESS_LIBRARY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library)
if(EXISTS ${CHESS_LIBRARY_ROOT}/include/chess.hpp OR EXISTS ${CHESS_LIBRARY_ROOT}/src/include/chess.hpp)
    message(STATUS "engine: found Disservin/chess-library at ${CHESS_LIBRARY_ROOT}")
    add_library(chess_library_header INTERFACE)
    # chess-library layout varies; try both.
    if(EXISTS ${CHESS_LIBRARY_ROOT}/include/chess.hpp)
        target_include_directories(chess_library_header INTERFACE ${CHESS_LIBRARY_ROOT}/include)
    else()
        target_include_directories(chess_library_header INTERFACE ${CHESS_LIBRARY_ROOT}/src/include)
    endif()
else()
    message(WARNING "engine: chess-library submodule not found; Chess engine targets will be skipped")
endif()

# cshogi C++ core — we vendor a thin subset under engine/third_party/cshogi_core
# (populated by third_party/cshogi-vendor.sh). If absent, shogi targets are skipped.
set(CSHOGI_CORE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cshogi_core)
if(EXISTS ${CSHOGI_CORE_ROOT}/CMakeLists.txt)
    add_subdirectory(${CSHOGI_CORE_ROOT})
else()
    message(STATUS "engine: cshogi core not vendored yet (run engine/third_party/vendor_cshogi.sh)")
endif()

# --------------- MLX-C (optional) ---------------
if(CHESS_MLX_WITH_MLX)
    set(MLX_C_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx-c)
    if(EXISTS ${MLX_C_ROOT}/CMakeLists.txt)
        # MLX options — turn off expensive sub-builds we don't need.
        set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        # If the Metal toolchain isn't installed, turning off metal lets us
        # build a CPU-only backend.  The engine can still run inference (just
        # slower).  Users can install the toolchain via:
        #   xcodebuild -downloadComponent MetalToolchain
        if(NOT DEFINED CHESS_MLX_BUILD_METAL)
            set(CHESS_MLX_BUILD_METAL AUTO CACHE STRING
                "Whether MLX should build the metal (GPU) backend (ON/OFF/AUTO)")
        endif()
        if(CHESS_MLX_BUILD_METAL STREQUAL "OFF")
            set(MLX_BUILD_METAL OFF CACHE BOOL "" FORCE)
            message(STATUS "engine: MLX metal backend DISABLED (CPU-only inference)")
        endif()
        add_subdirectory(${MLX_C_ROOT} EXCLUDE_FROM_ALL)
    else()
        message(WARNING "engine: mlx-c submodule not found; NN backend will use stub (random policy)")
        set(CHESS_MLX_WITH_MLX OFF)
    endif()
endif()

# --------------- nlohmann/json (vendored single-header, MIT) ---------------
# Available at engine/third_party/nlohmann_json/json.hpp
add_library(nlohmann_json_header INTERFACE)
target_include_directories(nlohmann_json_header INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# --------------- Core library (shared by chess + shogi) ---------------
add_library(chess_mlx_core STATIC
    src/core/position_stub.cpp
)
target_include_directories(chess_mlx_core PUBLIC include)
target_link_libraries(chess_mlx_core PUBLIC chess_mlx_warnings nlohmann_json_header)

# --------------- Chess traits ---------------
if(TARGET chess_library_header)
    add_library(chess_mlx_chess STATIC
        src/chess/chess_traits.cpp
    )
    target_link_libraries(chess_mlx_chess
        PUBLIC chess_mlx_core chess_library_header
        PRIVATE chess_mlx_warnings
    )
endif()

# --------------- Shogi traits ---------------
if(TARGET cshogi_core)
    add_library(chess_mlx_shogi STATIC
        src/shogi/shogi_traits.cpp
    )
    target_link_libraries(chess_mlx_shogi
        PUBLIC chess_mlx_core cshogi_core
        PRIVATE chess_mlx_warnings
    )
endif()

# --------------- Search (MCTS + multi-ponder) ---------------
add_library(chess_mlx_search STATIC
    src/search/mcts.cpp
    src/search/multi_ponder.cpp
)
target_link_libraries(chess_mlx_search
    PUBLIC chess_mlx_core
    PRIVATE chess_mlx_warnings
)

# --------------- NN backend ---------------
add_library(chess_mlx_nn STATIC
    src/nn/backend_stub.cpp
    src/nn/backend_factory.cpp
    src/nn/batcher.cpp
    src/nn/mlx_backend.cpp
    src/nn/lc0_backend.cpp
)
target_link_libraries(chess_mlx_nn
    PUBLIC chess_mlx_core
    PRIVATE chess_mlx_warnings
)
if(CHESS_MLX_WITH_MLX AND TARGET mlxc)
    # Note: the mlx-c upstream library target is called `mlxc` (no hyphen).
    target_link_libraries(chess_mlx_nn PUBLIC mlxc)
    target_compile_definitions(chess_mlx_nn PUBLIC CHESS_MLX_HAS_MLX=1)
    target_include_directories(chess_mlx_nn PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx-c)
    message(STATUS "engine: linking against mlxc (MLX-C API)")
else()
    message(STATUS "engine: mlx-c target 'mlxc' not found; NN backend will use stub")
endif()

# ONNX Runtime (LC0 backend)
if(CHESS_MLX_WITH_ONNX AND OnnxRuntime_FOUND)
    target_include_directories(chess_mlx_nn PUBLIC ${OnnxRuntime_INCLUDE_DIRS})
    # Link with full path or add rpath to avoid "library not found" at link time
    target_link_libraries(chess_mlx_nn PUBLIC ${OnnxRuntime_LIBRARIES})
    if(DEFINED ONNXRUNTIME_LIB_DIR)
        target_link_directories(chess_mlx_nn PUBLIC ${ONNXRUNTIME_LIB_DIR})
    endif()
    target_compile_definitions(chess_mlx_nn PUBLIC CHESS_MLX_HAS_ONNX=1)
    message(STATUS "engine: LC0/ONNX backend compiled in")
else()
    message(STATUS "engine: LC0/ONNX backend NOT compiled (CHESS_MLX_WITH_ONNX=${CHESS_MLX_WITH_ONNX})")
endif()

# --------------- Protocol layer ---------------
add_library(chess_mlx_protocol STATIC
    src/protocol/uci_loop.cpp
    src/protocol/usi_loop.cpp
    src/protocol/json_rpc_ext.cpp
)
target_link_libraries(chess_mlx_protocol
    PUBLIC chess_mlx_core chess_mlx_search chess_mlx_nn
    PRIVATE chess_mlx_warnings
)

# --------------- Main executables ---------------
# Chess engine binary (UCI)
if(TARGET chess_mlx_chess)
    add_executable(chess_engine src/chess/main_chess.cpp)
    target_link_libraries(chess_engine PRIVATE
        chess_mlx_chess chess_mlx_search chess_mlx_nn chess_mlx_protocol
    )
    set_target_properties(chess_engine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/bin
    )
endif()

# Shogi engine binary (USI)
if(TARGET chess_mlx_shogi)
    add_executable(shogi_engine src/shogi/main_shogi.cpp)
    target_link_libraries(shogi_engine PRIVATE
        chess_mlx_shogi chess_mlx_search chess_mlx_nn chess_mlx_protocol
    )
    set_target_properties(shogi_engine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/bin
    )
endif()

# --------------- Tests ---------------
if(CHESS_MLX_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(CHESS_MLX_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
