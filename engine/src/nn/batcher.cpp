// SPDX-License-Identifier: MIT
// engine/src/nn/batcher.cpp

#include "nn/batcher.hpp"

#include <chrono>
#include <utility>

namespace chess_mlx::nn {

// ---------------------------------------------------------------------------
// Owning constructor
// ---------------------------------------------------------------------------
Batcher::Batcher(std::shared_ptr<NNBackend> backend, BatcherConfig cfg)
    : shared_(nullptr), backend_(std::move(backend)), cfg_(cfg) {
    dispatcher_ = std::thread(&Batcher::dispatcher_loop, this);
}

// ---------------------------------------------------------------------------
// Non-owning ("shared delegate") constructor — just stores the pointer.
// No dispatcher thread is launched; submit() calls shared_->submit().
// ---------------------------------------------------------------------------
Batcher::Batcher(Batcher* shared) noexcept
    : shared_(shared) {}

Batcher::~Batcher() {
    if (shared_ == nullptr) {
        shutdown();
    }
}

std::future<NNOutput> Batcher::submit(std::vector<float> input_tensor) {
    // Delegate to the shared owning batcher if this is a non-owning instance.
    if (shared_ != nullptr) {
        return shared_->submit(std::move(input_tensor));
    }

    Request req;
    req.input = std::move(input_tensor);
    auto fut = req.promise.get_future();

    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push_back(std::move(req));
        total_requests_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_one();
    return fut;
}

NNOutput Batcher::evaluate_direct(const std::vector<float>& input_tensor) {
    if (shared_ != nullptr) {
        return shared_->evaluate_direct(input_tensor);
    }
    total_requests_.fetch_add(1, std::memory_order_relaxed);
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    return backend_->evaluate(input_tensor);
}

void Batcher::shutdown() {
    if (shared_ != nullptr) return;  // non-owning — nothing to shut down

    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    if (dispatcher_.joinable()) dispatcher_.join();

    // Drain remaining requests with exceptions so no one hangs on a future.
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& req : queue_) {
        try {
            req.promise.set_exception(
                std::make_exception_ptr(std::runtime_error("Batcher shut down")));
        } catch (...) {
            // promise already satisfied; ignore
        }
    }
    queue_.clear();
}

void Batcher::dispatcher_loop() {
    while (true) {
        std::vector<Request> batch;

        {
            std::unique_lock<std::mutex> lk(mtx_);

            // Wait for at least one request, or shutdown.
            cv_.wait(lk, [this]() {
                return stop_.load() || !queue_.empty();
            });

            if (stop_.load() && queue_.empty()) return;

            // If we have fewer than max_batch_size but >= 1, wait briefly
            // for more to accumulate (up to flush_timeout_us).
            if (queue_.size() < cfg_.max_batch_size) {
                cv_.wait_for(lk,
                    std::chrono::microseconds(cfg_.flush_timeout_us),
                    [this]() {
                        return stop_.load()
                            || queue_.size() >= cfg_.max_batch_size;
                    });
            }

            if (stop_.load() && queue_.empty()) return;

            const std::size_t take =
                std::min(queue_.size(), cfg_.max_batch_size);
            batch.reserve(take);
            for (std::size_t i = 0; i < take; ++i) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }

        if (batch.empty()) continue;

        // Assemble inputs and run the batch.
        std::vector<std::vector<float>> inputs;
        inputs.reserve(batch.size());
        for (auto& req : batch) inputs.push_back(std::move(req.input));

        std::vector<NNOutput> outputs;
        try {
            outputs = backend_->evaluate_batch(inputs);
        } catch (...) {
            auto ep = std::current_exception();
            for (auto& req : batch) {
                try { req.promise.set_exception(ep); } catch (...) {}
            }
            continue;
        }

        total_batches_.fetch_add(1, std::memory_order_relaxed);

        // Fulfil each promise.
        for (std::size_t i = 0; i < batch.size(); ++i) {
            try {
                if (i < outputs.size()) {
                    batch[i].promise.set_value(std::move(outputs[i]));
                } else {
                    batch[i].promise.set_exception(
                        std::make_exception_ptr(
                            std::runtime_error("batch output size mismatch")));
                }
            } catch (...) {
                // promise already satisfied
            }
        }
    }
}

} // namespace chess_mlx::nn
