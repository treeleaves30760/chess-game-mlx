// SPDX-License-Identifier: MIT
// engine/tests/test_lc0_backend.cpp
//
// Integration tests for Lc0Backend:
//   1. Load BT4.onnx and run a forward pass on startpos
//   2. Check policy argmax gives a canonical opening move
//   3. Golden data validation: C++ encoder output vs lc0_bt4_inputs.json
//   4. Golden data validation: ONNX forward pass vs lc0_bt4_outputs.json

#include "nn/lc0_backend.hpp"
#include "chess/lc0_chess_traits.hpp"
#include "chess/lc0_encoding.hpp"
#include "chess/lc0_move_index.hpp"

#include <chess.hpp>
#include <nlohmann_json/json.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#ifdef CHESS_MLX_HAS_ONNX

using nlohmann::json;
using namespace chess_mlx;
using namespace chess_mlx::chess;
using namespace chess_mlx::chess::lc0;

// ============================================================================
// Helper: paths
// ============================================================================

// BT4 ONNX path relative to project root (set at compile time or via env).
static std::string bt4_path() {
    const char* env = std::getenv("BT4_ONNX_PATH");
    if (env && *env) return std::string(env);
    // Default relative to the binary's location.
    return "/Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx/data/lc0_nets/BT4.onnx";
}

static std::string golden_inputs_path() {
    const char* env = std::getenv("LC0_GOLDEN_INPUTS");
    if (env && *env) return std::string(env);
    return "/Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx/shared/golden_data/lc0_bt4_inputs.json";
}

static std::string golden_outputs_path() {
    const char* env = std::getenv("LC0_GOLDEN_OUTPUTS");
    if (env && *env) return std::string(env);
    return "/Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx/shared/golden_data/lc0_bt4_outputs.json";
}

// ============================================================================
// Helper: encode one position
// ============================================================================

static std::vector<float> encode_fen(const std::string& fen) {
    ChessPosition pos(fen);
    std::vector<float> out(64 * 112, 0.0f);
    static const std::vector<ChessPosition> kNoHistory;
    lc0::encode_nn(pos, kNoHistory, out.data());
    return out;
}

// ============================================================================
// Helper: softmax
// ============================================================================

static std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - mx);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    return probs;
}

// ============================================================================
// Test: load BT4 and run startpos
// ============================================================================

TEST(Lc0Backend, LoadAndEvaluateStartpos) {
    nn::Lc0Backend backend(bt4_path());
    EXPECT_EQ(backend.name(), "lc0-onnx");
    EXPECT_EQ(backend.input_size(), 64u * 112u);
    EXPECT_EQ(backend.policy_size(), 1858u);

    auto input = encode_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto out = backend.evaluate(input);

    EXPECT_EQ(out.policy.size(), 1858u);

    // Value should be near 0 (equal position)
    EXPECT_NEAR(out.value, 0.0f, 0.15f)
        << "Startpos value should be near 0, got " << out.value;

    // Policy argmax should be a canonical opening move.
    // Top policy slots for startpos: 293 (d2d4), 322 (e2e4), 159 (g1f3)
    int argmax = static_cast<int>(
        std::max_element(out.policy.begin(), out.policy.end()) - out.policy.begin());

    // Check the argmax move is a reasonable opening move
    bool canonical = (argmax == 293 || argmax == 322 || argmax == 159  // d4, e4, Nf3
                   || argmax == 518 || argmax == 662 || argmax == 855); // other strong moves
    EXPECT_TRUE(canonical)
        << "Startpos policy argmax=" << argmax
        << " (" << kLc0MoveStrs[static_cast<std::size_t>(argmax)] << ")"
        << " is not a canonical opening move";

    // Moves-left should be positive and sane (10..300 plies)
    EXPECT_GT(out.moves_left, 10.0f);
    EXPECT_LT(out.moves_left, 500.0f);
}

// ============================================================================
// Test: Golden encoder data validation
// ============================================================================

TEST(Lc0Backend, GoldenEncoderValidation) {
    // Load golden inputs JSON
    std::ifstream f(golden_inputs_path());
    if (!f.is_open()) {
        GTEST_SKIP() << "Golden inputs JSON not found: " << golden_inputs_path();
    }
    json j;
    f >> j;

    const auto& cases = j["cases"];
    EXPECT_GT(cases.size(), 0u);

    int case_idx = 0;
    for (const auto& c : cases) {
        const std::string fen = c["fen"].get<std::string>();
        const std::string name = c["name"].get<std::string>();
        const auto& expected_flat = c["planes_flat"];  // [112 * 64] NCHW layout

        // Our encoding: square-major [64 * 112], our[sq * 112 + plane]
        auto our_planes = encode_fen(fen);
        ASSERT_EQ(our_planes.size(), 64u * 112u)
            << "Case " << name << ": wrong output size";

        // Golden is NCHW: golden[plane * 64 + sq]
        std::vector<float> golden(expected_flat.begin(), expected_flat.end());
        ASSERT_EQ(golden.size(), 64u * 112u)
            << "Case " << name << ": wrong golden size";

        // Compare by iterating (plane, sq) pairs and cross-indexing layouts.
        // golden[plane * 64 + sq] == our_planes[sq * 112 + plane]
        float max_diff = 0.0f;
        std::size_t worst_plane = 0, worst_sq = 0;
        for (std::size_t plane = 0; plane < 112u; ++plane) {
            for (std::size_t sq = 0; sq < 64u; ++sq) {
                float g = golden[plane * 64u + sq];
                float o = our_planes[sq * 112u + plane];
                float diff = std::abs(o - g);
                if (diff > max_diff) {
                    max_diff = diff;
                    worst_plane = plane;
                    worst_sq = sq;
                }
            }
        }

        EXPECT_LE(max_diff, 0.01f)
            << "Case " << name
            << ": encoder mismatch at plane=" << worst_plane
            << " sq=" << worst_sq
            << " our=" << our_planes[worst_sq * 112u + worst_plane]
            << " golden=" << golden[worst_plane * 64u + worst_sq];

        ++case_idx;
    }
}

// ============================================================================
// Test: Golden outputs validation (ONNX forward pass)
// ============================================================================

TEST(Lc0Backend, GoldenOutputValidation) {
    std::ifstream f(golden_outputs_path());
    if (!f.is_open()) {
        GTEST_SKIP() << "Golden outputs JSON not found: " << golden_outputs_path();
    }
    json j;
    f >> j;

    nn::Lc0Backend backend(bt4_path());

    const float policy_tol = j.value("tolerance_policy_prob", 0.005f);
    const float wdl_tol    = j.value("tolerance_wdl", 0.01f);

    const auto& cases = j["cases"];
    for (const auto& c : cases) {
        const std::string name = c["name"].get<std::string>();
        const std::string fen  = c["fen"].get<std::string>();

        // Golden policy logits [1858]
        std::vector<float> golden_logits(c["policy_logits"].begin(),
                                          c["policy_logits"].end());
        ASSERT_EQ(golden_logits.size(), 1858u);

        // Golden WDL (already softmaxed in the JSON)
        std::vector<float> golden_wdl(c["wdl"].begin(), c["wdl"].end());
        ASSERT_EQ(golden_wdl.size(), 3u);
        const float golden_value = golden_wdl[0] - golden_wdl[2];

        // Run our backend
        auto input = encode_fen(fen);
        auto out = backend.evaluate(input);
        ASSERT_EQ(out.policy.size(), 1858u);

        // Compare top-3 policy slots by probability
        auto our_probs    = softmax(out.policy);
        auto golden_probs = softmax(golden_logits);

        // Sort by golden prob to get top-3
        std::vector<int> order(1858);
        for (int i = 0; i < 1858; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return golden_probs[a] > golden_probs[b];
        });

        for (int k = 0; k < 3; ++k) {
            int slot = order[k];
            float our_p    = our_probs[static_cast<std::size_t>(slot)];
            float golden_p = golden_probs[static_cast<std::size_t>(slot)];
            EXPECT_NEAR(our_p, golden_p, policy_tol)
                << "Case " << name << ": top-" << (k+1)
                << " policy slot " << slot << " ("
                << kLc0MoveStrs[static_cast<std::size_t>(slot)] << ")"
                << " our_prob=" << our_p
                << " golden=" << golden_p;
        }

        // Compare value (W - L)
        EXPECT_NEAR(out.value, golden_value, wdl_tol)
            << "Case " << name
            << ": value our=" << out.value
            << " golden=" << golden_value;
    }
}

#else  // CHESS_MLX_HAS_ONNX not defined

TEST(Lc0Backend, Skipped) {
    GTEST_SKIP() << "ONNX Runtime not compiled in (CHESS_MLX_WITH_ONNX=OFF)";
}

#endif  // CHESS_MLX_HAS_ONNX
