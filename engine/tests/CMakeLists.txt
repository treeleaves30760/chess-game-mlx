# Tests use GoogleTest fetched via FetchContent (no vendoring).
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# --------------- Core tests ---------------
add_executable(test_zobrist test_zobrist.cpp)
target_link_libraries(test_zobrist PRIVATE chess_mlx_core GTest::gtest_main)
gtest_discover_tests(test_zobrist)

# --------------- Chess tests ---------------
if(TARGET chess_mlx_chess)
    add_executable(test_perft_chess test_perft_chess.cpp)
    target_link_libraries(test_perft_chess PRIVATE chess_mlx_chess GTest::gtest_main)
    gtest_discover_tests(test_perft_chess PROPERTIES TIMEOUT 120)

    add_executable(test_encoding_chess test_encoding_chess.cpp)
    target_link_libraries(test_encoding_chess PRIVATE chess_mlx_chess GTest::gtest_main)
    gtest_discover_tests(test_encoding_chess)

    add_executable(test_lc0_encoding test_lc0_encoding.cpp)
    target_link_libraries(test_lc0_encoding PRIVATE chess_mlx_chess GTest::gtest_main)
    gtest_discover_tests(test_lc0_encoding)

    add_executable(test_chess_compact_1858 test_chess_compact_1858.cpp)
    target_link_libraries(test_chess_compact_1858 PRIVATE chess_mlx_chess GTest::gtest_main)
    gtest_discover_tests(test_chess_compact_1858)
endif()

# --------------- Shogi tests ---------------
if(TARGET chess_mlx_shogi)
    add_executable(test_perft_shogi test_perft_shogi.cpp)
    target_link_libraries(test_perft_shogi PRIVATE chess_mlx_shogi GTest::gtest_main)
    gtest_discover_tests(test_perft_shogi PROPERTIES TIMEOUT 120)
endif()

# --------------- Search tests (chess, MCTS needs ChessTraits) ---------------
if(TARGET chess_mlx_chess)
    add_executable(test_mcts test_mcts.cpp)
    target_link_libraries(test_mcts PRIVATE
        chess_mlx_chess chess_mlx_search chess_mlx_nn GTest::gtest_main)
    gtest_discover_tests(test_mcts PROPERTIES TIMEOUT 60)
endif()

# --------------- NN backend tests ---------------
add_executable(test_nn_backend test_nn_backend.cpp)
target_link_libraries(test_nn_backend PRIVATE
    chess_mlx_nn GTest::gtest_main)
gtest_discover_tests(test_nn_backend PROPERTIES TIMEOUT 60)

# --------------- Multi-ponder tests (chess, Phase 6) ---------------
if(TARGET chess_mlx_chess)
    add_executable(test_multi_ponder test_multi_ponder.cpp)
    target_link_libraries(test_multi_ponder PRIVATE
        chess_mlx_chess chess_mlx_search chess_mlx_nn chess_mlx_protocol
        GTest::gtest_main)
    gtest_discover_tests(test_multi_ponder PROPERTIES TIMEOUT 60)
endif()

# --------------- LC0 move-index tests ---------------
if(TARGET chess_mlx_chess)
    add_executable(test_lc0_move_index test_lc0_move_index.cpp)
    target_link_libraries(test_lc0_move_index PRIVATE
        chess_mlx_chess chess_mlx_nn GTest::gtest_main)
    gtest_discover_tests(test_lc0_move_index)
endif()

# --------------- LC0 backend tests (requires BT4.onnx and ONNX Runtime) ---------------
if(TARGET chess_mlx_chess AND CHESS_MLX_WITH_ONNX AND OnnxRuntime_FOUND)
    add_executable(test_lc0_backend test_lc0_backend.cpp)
    target_link_libraries(test_lc0_backend PRIVATE
        chess_mlx_chess chess_mlx_nn GTest::gtest_main)
    # These tests load BT4.onnx and run ONNX inference — allow up to 5 minutes.
    gtest_discover_tests(test_lc0_backend PROPERTIES TIMEOUT 300)
endif()
