// SPDX-License-Identifier: MIT
// engine/tests/test_nn_backend.cpp
//
// NN backend tests: verify the StubBackend + (if available) MlxBackend.

#include <gtest/gtest.h>

#include "nn/backend.hpp"
#include "nn/backend_factory.hpp"
#include "nn/batcher.hpp"
#include "nn/stub_backend.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace chess_mlx::nn;

// ----------------------------------------------------------------------------
// StubBackend: uniform policy, value 0, moves_left 50.
// ----------------------------------------------------------------------------
TEST(StubBackend, ChessOutputShape) {
    StubBackend b(4672, 64 * 19);
    std::vector<float> in(64 * 19, 0.0f);
    auto out = b.evaluate(in);
    EXPECT_EQ(out.policy.size(), 4672u);
    EXPECT_EQ(out.value, 0.0f);
    EXPECT_FLOAT_EQ(out.moves_left, 50.0f);
    EXPECT_EQ(b.name(), std::string("stub"));
    EXPECT_EQ(b.input_size(), 64u * 19u);
    EXPECT_EQ(b.policy_size(), 4672u);
}

TEST(StubBackend, ShogiOutputShape) {
    StubBackend b(2187, 81 * 90);
    std::vector<float> in(81 * 90, 0.5f);
    auto out = b.evaluate(in);
    EXPECT_EQ(out.policy.size(), 2187u);
    EXPECT_EQ(out.value, 0.0f);
}

TEST(StubBackend, Batch) {
    StubBackend b(4672, 64 * 19);
    std::vector<std::vector<float>> inputs(4, std::vector<float>(64 * 19, 0.0f));
    auto outs = b.evaluate_batch(inputs);
    EXPECT_EQ(outs.size(), 4u);
    for (auto& o : outs) EXPECT_EQ(o.policy.size(), 4672u);
}

// ----------------------------------------------------------------------------
// Batcher: submit several async requests, all futures fulfilled.
// ----------------------------------------------------------------------------
TEST(Batcher, AsyncSubmitFulfilled) {
    auto backend = std::make_shared<StubBackend>(4672, 64 * 19);
    BatcherConfig cfg;
    cfg.max_batch_size = 4;
    cfg.flush_timeout_us = 100;
    Batcher b(backend, cfg);

    constexpr int kN = 16;
    std::vector<std::future<NNOutput>> futs;
    futs.reserve(kN);
    for (int i = 0; i < kN; ++i) {
        futs.push_back(b.submit(std::vector<float>(64 * 19, static_cast<float>(i))));
    }
    for (auto& f : futs) {
        auto out = f.get();
        EXPECT_EQ(out.policy.size(), 4672u);
    }
    b.shutdown();
    EXPECT_GE(b.total_requests(), static_cast<std::uint64_t>(kN));
}

TEST(Batcher, ConcurrentSubmit) {
    auto backend = std::make_shared<StubBackend>(4672, 64 * 19);
    Batcher b(backend);

    std::vector<std::thread> threads;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 10;

    std::atomic<int> success{0};
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kPerThread; ++i) {
                auto fut = b.submit(std::vector<float>(64 * 19, 0.0f));
                auto out = fut.get();
                if (out.policy.size() == 4672u) success.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) t.join();
    b.shutdown();
    EXPECT_EQ(success.load(), kThreads * kPerThread);
}

// ----------------------------------------------------------------------------
// Factory: make_nn_backend with empty weights returns StubBackend.
// ----------------------------------------------------------------------------
TEST(BackendFactory, EmptyWeightsReturnsStub) {
    BackendConfig cfg;
    cfg.game = Game::Chess;
    cfg.weights_path = "";
    auto backend = make_nn_backend(cfg);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->name(), "stub");
    EXPECT_EQ(backend->policy_size(), 4672u);
}

TEST(BackendFactory, ForceStub) {
    BackendConfig cfg;
    cfg.game = Game::Shogi;
    cfg.force_stub = true;
    auto backend = make_nn_backend(cfg);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->name(), "stub");
    EXPECT_EQ(backend->policy_size(), 2187u);
}

TEST(BackendFactory, NonexistentWeightsFallsBackToStub) {
    BackendConfig cfg;
    cfg.game = Game::Chess;
    cfg.weights_path = "/tmp/this_file_does_not_exist_xyz_789.safetensors";
    auto backend = make_nn_backend(cfg);
    ASSERT_NE(backend, nullptr);
    // Either MLX or stub — either way, the call must succeed and return
    // something of the right shape.
    EXPECT_EQ(backend->policy_size(), 4672u);
}

// ----------------------------------------------------------------------------
// MLX backend forward pass (only runs if a dummy checkpoint is present).
// ----------------------------------------------------------------------------
TEST(MlxBackend, DummyCheckpointIfAvailable) {
#ifdef CHESS_MLX_HAS_MLX
    const char* path = "/tmp/dummy_chess.safetensors";
    std::ifstream f(path);
    if (!f.good()) {
        GTEST_SKIP() << "no dummy checkpoint at " << path
                     << " (run training/scripts/export_dummy.py to create)";
    }
    f.close();

    BackendConfig cfg;
    cfg.game = Game::Chess;
    cfg.weights_path = path;
    auto backend = make_nn_backend(cfg);
    ASSERT_NE(backend, nullptr);

    std::vector<float> in(64 * 19, 0.0f);
    // Set a few squares to 1 (doesn't matter for shape check).
    in[0] = 1.0f; in[100] = 1.0f;
    auto out = backend->evaluate(in);
    EXPECT_EQ(out.policy.size(), 4672u);
    EXPECT_GE(out.value, -1.0f);
    EXPECT_LE(out.value,  1.0f);
    EXPECT_GE(out.moves_left, 0.0f);
#else
    GTEST_SKIP() << "CHESS_MLX_HAS_MLX not defined";
#endif
}

TEST(BackendFactory, MlxAvailableFlag) {
#ifdef CHESS_MLX_HAS_MLX
    EXPECT_TRUE(mlx_available());
#else
    EXPECT_FALSE(mlx_available());
#endif
}
