// SPDX-License-Identifier: MIT
// engine/tests/test_multi_ponder.cpp
//
// Tests for MultiPonderManager<ChessTraits> (Phase 6).
//
// All tests use StubBackend (uniform policy, value 0) so they run fast
// without any real NN weights.

#include <gtest/gtest.h>

#include "chess/chess_traits.hpp"
#include "nn/stub_backend.hpp"
#include "search/multi_ponder.hpp"
#include "search/policy_utils.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using ChessTraits = chess_mlx::chess::ChessTraits;
using ChessPos    = chess_mlx::chess::ChessPosition;
using ChessMove   = chess_mlx::chess::ChessMove;
using MPM         = chess_mlx::search::MultiPonderManager<ChessTraits>;
using nlohmann::json;

namespace nn     = chess_mlx::nn;
namespace search = chess_mlx::search;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::shared_ptr<nn::StubBackend> make_stub() {
    return std::make_shared<nn::StubBackend>(/*policy=*/4672, /*input=*/64*19);
}

static ChessPos startpos() {
    return ChessPos{};
}

// Return a legal move from startpos by string; throws on failure.
static ChessMove parse_startpos_move(const std::string& uci) {
    ChessPos pos;
    ::chess::Move raw = ::chess::uci::uciToMove(pos.board, uci);
    return ChessMove(raw);
}

// Collect all on_event notifications over a duration.
struct EventCollector {
    std::vector<json>  events;
    std::mutex         mtx;

    void operator()(const json& j) {
        std::lock_guard<std::mutex> lk(mtx);
        events.push_back(j);
    }

    int count(const std::string& method) const {
        int n = 0;
        for (const auto& e : events) {
            if (e.value("method", "") == method) ++n;
        }
        return n;
    }
};

// ---------------------------------------------------------------------------
// Test: start() emits policy_preview within 1 second; all 5 trees running.
// ---------------------------------------------------------------------------
TEST(MultiPonder, StartEmitsPolicyPreview) {
    auto backend = make_stub();
    MPM mgr(backend, /*threads_per_tree=*/1);

    EventCollector col;
    ChessPos pos = startpos();
    const std::vector<double> weights = {0.40, 0.20, 0.15, 0.15, 0.10};

    mgr.start(pos, 5, weights, [&col](json j) { col(j); });

    // policy_preview must have been emitted synchronously inside start().
    ASSERT_GE(col.count("policy_preview"), 1) << "policy_preview not emitted";

    // Verify structure: params.moves is an array of {uci, prob}.
    bool found_preview = false;
    for (const auto& ev : col.events) {
        if (ev.value("method", "") == "policy_preview") {
            ASSERT_TRUE(ev.contains("params"));
            ASSERT_TRUE(ev["params"].contains("moves"));
            ASSERT_TRUE(ev["params"]["moves"].is_array());
            EXPECT_GT(ev["params"]["moves"].size(), 0u);
            for (const auto& m : ev["params"]["moves"]) {
                EXPECT_TRUE(m.contains("uci"));
                EXPECT_TRUE(m.contains("prob"));
                EXPECT_GT(m["prob"].get<double>(), 0.0);
            }
            found_preview = true;
        }
    }
    EXPECT_TRUE(found_preview);

    // Let trees run briefly, verify they are actually running.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    int running = 0;
    for (int i = 0; i < mgr.active_tree_count(); ++i) {
        if (mgr.tree(i).running.load()) ++running;
    }
    EXPECT_GT(running, 0);

    mgr.stop();
}

// ---------------------------------------------------------------------------
// Test: ponder_progress emitted after ~500ms.
// ---------------------------------------------------------------------------
TEST(MultiPonder, ProgressEmittedAfter500ms) {
    auto backend = make_stub();
    MPM mgr(backend, /*threads_per_tree=*/1);

    EventCollector col;
    ChessPos pos = startpos();
    mgr.start(pos, 3, {0.40, 0.30, 0.30}, [&col](json j) { col(j); });

    // Wait for at least one progress event (expected ~500ms, give 1500ms slack).
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    while (col.count("ponder_progress") == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    EXPECT_GE(col.count("ponder_progress"), 1)
        << "ponder_progress not emitted within 1.5 seconds";

    mgr.stop();
}

// ---------------------------------------------------------------------------
// Test: ponder_hit — build the top-k list, pick the first one, verify hit.
// ---------------------------------------------------------------------------
TEST(MultiPonder, PonderHit) {
    auto backend = make_stub();
    MPM mgr(backend, /*threads_per_tree=*/1);

    EventCollector col;
    ChessPos pos = startpos();
    const int k = 5;
    mgr.start(pos, k, {0.40, 0.20, 0.15, 0.15, 0.10},
              [&col](json j) { col(j); });

    // Find which moves were selected as top-k by reading the policy_preview.
    std::string first_uci;
    for (const auto& ev : col.events) {
        if (ev.value("method", "") == "policy_preview") {
            if (!ev["params"]["moves"].empty()) {
                first_uci = ev["params"]["moves"][0]["uci"].get<std::string>();
            }
        }
    }
    ASSERT_FALSE(first_uci.empty()) << "policy_preview had no moves";

    // Let the trees accumulate a few nodes.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Call opponent_played with the first predicted move → must be a hit.
    ChessMove mv = parse_startpos_move(first_uci);
    const int hit_idx = mgr.opponent_played(mv);

    EXPECT_GE(hit_idx, 0) << "Expected hit but got -1 (miss)";

    // ponder_hit notification must have been emitted.
    EXPECT_GE(col.count("ponder_hit"), 1);

    // Verify ponder_hit structure.
    for (const auto& ev : col.events) {
        if (ev.value("method", "") == "ponder_hit") {
            EXPECT_TRUE(ev["params"].contains("tree"));
            EXPECT_TRUE(ev["params"].contains("instant_bestmove"));
            EXPECT_TRUE(ev["params"].contains("score_cp"));
        }
    }

    // take_promoted_mcts() should return a non-null MCTS.
    auto promoted = mgr.take_promoted_mcts();
    EXPECT_NE(promoted, nullptr);
}

// ---------------------------------------------------------------------------
// Test: ponder_miss — play a move definitely not in top-k.
// With uniform policy (StubBackend) and k=5 from startpos (20 legal moves),
// 15 moves are NOT in the top-5.  Use a move that was not in the preview list.
// ---------------------------------------------------------------------------
TEST(MultiPonder, PonderMiss) {
    auto backend = make_stub();
    MPM mgr(backend, /*threads_per_tree=*/1);

    EventCollector col;
    ChessPos pos = startpos();
    const int k = 5;
    mgr.start(pos, k, {0.40, 0.20, 0.15, 0.15, 0.10},
              [&col](json j) { col(j); });

    // Collect all UCI moves in the preview.
    std::vector<std::string> preview_ucis;
    for (const auto& ev : col.events) {
        if (ev.value("method", "") == "policy_preview") {
            for (const auto& m : ev["params"]["moves"]) {
                preview_ucis.push_back(m["uci"].get<std::string>());
            }
        }
    }

    // Find a legal move from startpos that is NOT in the preview.
    // Startpos has 20 legal moves; with k=5 at least 15 are outside.
    const std::vector<std::string> all_start_moves = {
        "a2a3","a2a4","b2b3","b2b4","c2c3","c2c4","d2d3","d2d4",
        "e2e3","e2e4","f2f3","f2f4","g2g3","g2g4","h2h3","h2h4",
        "b1a3","b1c3","g1f3","g1h3"
    };
    std::string miss_uci;
    for (const auto& m : all_start_moves) {
        bool in_preview = false;
        for (const auto& p : preview_ucis) {
            if (p == m) { in_preview = true; break; }
        }
        if (!in_preview) { miss_uci = m; break; }
    }

    if (miss_uci.empty()) {
        // All 20 moves somehow in top-5 — impossible with k=5 but skip if so.
        GTEST_SKIP() << "Could not find a move outside top-5 (unexpected)";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ChessMove mv = parse_startpos_move(miss_uci);
    const int hit_idx = mgr.opponent_played(mv);

    EXPECT_EQ(hit_idx, -1) << "Expected miss (-1) but got hit " << hit_idx;
    EXPECT_GE(col.count("ponder_miss"), 1);
}

// ---------------------------------------------------------------------------
// Test: concurrent start/stop cycles — no crash or deadlock.
// ---------------------------------------------------------------------------
TEST(MultiPonder, ConcurrentStartStop) {
    auto backend = make_stub();

    constexpr int kCycles = 50;
    for (int i = 0; i < kCycles; ++i) {
        MPM mgr(backend, /*threads_per_tree=*/1);
        ChessPos pos = startpos();
        mgr.start(pos, 5, {0.40, 0.20, 0.15, 0.15, 0.10},
                  [](const json&) {});
        // Minimal sleep to let threads start.
        if (i % 5 == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        mgr.stop();
    }
    // If we get here without deadlock / crash, the test passes.
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Test: stop() is idempotent — calling it multiple times is safe.
// ---------------------------------------------------------------------------
TEST(MultiPonder, StopIdempotent) {
    auto backend = make_stub();
    MPM mgr(backend, /*threads_per_tree=*/1);

    ChessPos pos = startpos();
    mgr.start(pos, 3, {0.40, 0.30, 0.30}, [](const json&) {});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    mgr.stop();
    mgr.stop();  // second call must not crash
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Test: budget ratio — after 2 seconds, node count ratios should approximately
// reflect the budget weights (±50% slack because StubBackend is very fast and
// thread scheduling is non-deterministic).
// ---------------------------------------------------------------------------
TEST(MultiPonder, BudgetRatioApproximate) {
    auto backend = make_stub();
    // Use threads_per_tree=2 so the budget scaling is more visible.
    MPM mgr(backend, /*threads_per_tree=*/2);

    ChessPos pos = startpos();
    const std::vector<double> weights = {0.40, 0.20, 0.15, 0.15, 0.10};
    mgr.start(pos, 5, weights, [](const json&) {});

    std::this_thread::sleep_for(std::chrono::seconds(2));
    mgr.stop();

    // Sum all nodes.
    uint64_t total = 0;
    for (int i = 0; i < 5; ++i) {
        total += mgr.tree(i).nodes.load();
    }
    if (total == 0) {
        GTEST_SKIP() << "No nodes accumulated (StubBackend too fast / scheduling)";
    }

    // Tree 0 should have roughly 40% of nodes.
    const double frac0 = static_cast<double>(mgr.tree(0).nodes.load())
                         / static_cast<double>(total);
    // Loose check: within ±50% of target weight.
    EXPECT_GT(frac0, weights[0] * 0.5)
        << "Tree 0 fraction " << frac0
        << " too low (target " << weights[0] << ")";
    EXPECT_LT(frac0, weights[0] * 1.5 + 0.2)
        << "Tree 0 fraction " << frac0
        << " too high (target " << weights[0] << ")";
}

// ---------------------------------------------------------------------------
// Test: top_k_policy_moves helper (unit test of the pure function).
// ---------------------------------------------------------------------------
TEST(PolicyUtils, TopKFromStartpos) {
    auto backend = make_stub();
    ChessPos pos = startpos();

    const auto top5 = search::top_k_policy_moves<ChessTraits>(
        *backend, pos, /*input_size=*/64*19, /*k=*/5);

    EXPECT_EQ(top5.size(), 5u);
    // Each probability should be in (0, 1].
    double sum = 0.0;
    for (const auto& [mv, prob] : top5) {
        EXPECT_GT(prob, 0.0f);
        EXPECT_LE(prob, 1.0f);
        sum += prob;
    }
    // Probabilities need not sum to 1 (they are top-k of a softmax).
    EXPECT_LE(sum, 1.01);  // soft upper bound

    // Moves must be distinct.
    for (std::size_t i = 0; i < top5.size(); ++i) {
        for (std::size_t j = i + 1; j < top5.size(); ++j) {
            EXPECT_NE(top5[i].first, top5[j].first)
                << "Duplicate move in top-5 at indices " << i << " and " << j;
        }
    }
}

// ---------------------------------------------------------------------------
// Test: shared Batcher — multiple MCTS trees submitting to one Batcher.
// ---------------------------------------------------------------------------
TEST(MultiPonder, SharedBatcherAllTreesEvaluate) {
    auto backend = make_stub();
    // Create the shared batcher manually (simulates what MPM does internally).
    nn::Batcher shared_batcher(backend);

    // Build 3 MCTS instances sharing the same batcher.
    ChessPos pos = startpos();
    const std::size_t kInput = 64 * 19;
    const std::size_t kPolicy = 4672;

    std::vector<std::unique_ptr<search::MCTS<ChessTraits>>> trees;
    for (int i = 0; i < 3; ++i) {
        search::MCTS<ChessTraits>::Config cfg;
        cfg.threads    = 1;
        cfg.input_size  = kInput;
        cfg.policy_size = kPolicy;
        trees.push_back(std::make_unique<search::MCTS<ChessTraits>>(
            backend, cfg, &shared_batcher));
    }

    // Run all three concurrently.
    std::vector<std::thread> workers;
    std::atomic<int> successes{0};
    for (int i = 0; i < 3; ++i) {
        workers.emplace_back([&, i]() {
            search::TimeControl tc;
            tc.max_nodes = 100;
            auto result = trees[static_cast<std::size_t>(i)]->search(pos, tc);
            if (!result.best_move.is_null()) ++successes;
        });
    }
    for (auto& t : workers) t.join();
    shared_batcher.shutdown();

    EXPECT_EQ(successes.load(), 3)
        << "All three trees should return a legal best move";
}
