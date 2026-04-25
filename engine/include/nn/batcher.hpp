// SPDX-License-Identifier: MIT
// engine/include/nn/batcher.hpp
//
// Batcher — accumulates leaf-evaluation requests from multiple MCTS worker
// threads into one NN call.  Workers block on a per-request future until the
// batch dispatch returns their result.
//
// Design:
//   * Each worker constructs an `EvalRequest` holding its input tensor and a
//     std::promise<NNOutput>.
//   * `submit()` pushes the request onto a queue and either:
//       - wakes a dispatcher thread when the queue hits `batch_size`, or
//       - returns immediately, letting the timeout hit for smaller batches.
//   * A single dispatcher thread waits on a condvar, flushes the queue when
//     either the batch is full or `flush_timeout_us` passes since the oldest
//     request, calls `backend->evaluate_batch(...)`, and fulfils all promises.
//
// This layer is intentionally stateless about the game — it just funnels
// float32 tensors through.
//
// Phase 6 addition: Batcher can be constructed in "shared" mode by providing
// a raw pointer to an externally-owned Batcher.  In that mode the constructor
// that takes `Batcher* shared` stores a non-owning reference, and submit()
// delegates to the shared batcher.  This is how MultiPonderManager wires 5
// MCTS trees to a single GPU dispatch queue for cross-tree batching.

#pragma once

#include "nn/backend.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace chess_mlx::nn {

struct BatcherConfig {
    // Maximum batch size.  256 is good for multi-ponder (5 trees × ~50 leaves).
    std::size_t max_batch_size = 256;
    // If the queue has pending requests but hasn't reached max_batch_size,
    // flush after this many microseconds anyway.  Balances latency vs
    // batching benefit.
    //
    // Flush timeout in microseconds.  The batcher fires a batch when either
    // max_batch_size requests have accumulated or this timeout expires.
    //
    // 500µs is a good default for the pipelined GPU backend: after a batch
    // resolves, all worker threads backprop + select + submit new leaves within
    // ~100-200µs.  500µs comfortably captures the full batch while reducing
    // idle batcher wait from 2ms → 0.5ms (4× more batches/sec).
    // For CPU / stub backends the overhead is irrelevant because they
    // bypass the batcher entirely.
    std::size_t flush_timeout_us = 500;
};

class Batcher {
public:
    // Owning constructor — Batcher manages its own dispatcher thread.
    Batcher(std::shared_ptr<NNBackend> backend, BatcherConfig cfg = {});

    // Non-owning ("shared") constructor — submits delegate to `shared`.
    // The pointed-to Batcher must outlive this object.
    explicit Batcher(Batcher* shared) noexcept;

    ~Batcher();

    // Submit a tensor for evaluation.  Returns a future that will be set when
    // the batch fires.  Thread-safe.
    std::future<NNOutput> submit(std::vector<float> input_tensor);

    // Synchronously evaluate (bypasses batcher thread — used in single-thread
    // benchmark mode for minimum latency).
    NNOutput evaluate_direct(const std::vector<float>& input_tensor);

    // Stop the dispatcher thread and drain any pending requests.
    // No-op on a non-owning Batcher.
    void shutdown();

    // Observability.
    std::uint64_t total_batches () const { return total_batches_.load();  }
    std::uint64_t total_requests() const { return total_requests_.load(); }

    // True if this is a non-owning delegate to another Batcher.
    bool is_shared() const noexcept { return shared_ != nullptr; }

private:
    struct Request {
        std::vector<float>       input;
        std::promise<NNOutput>   promise;
    };

    void dispatcher_loop();

    // Non-null only when this is the shared delegate mode.
    Batcher*                    shared_{nullptr};

    std::shared_ptr<NNBackend>  backend_;
    BatcherConfig               cfg_;

    std::mutex                  mtx_;
    std::condition_variable     cv_;
    std::deque<Request>         queue_;
    std::atomic<bool>           stop_{false};

    std::thread                 dispatcher_;

    std::atomic<std::uint64_t>  total_batches_{0};
    std::atomic<std::uint64_t>  total_requests_{0};
};

} // namespace chess_mlx::nn
