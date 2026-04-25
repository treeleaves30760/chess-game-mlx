// SPDX-License-Identifier: MIT
// engine/include/nn/lc0_backend.hpp
//
// Lc0Backend — NNBackend that loads BT4.onnx (or any LC0 ONNX export) via
// ONNX Runtime and maps the 1858-slot policy output to the engine's 4672-slot
// space so that MCTS can use it unchanged.
//
// Thread safety: evaluate() and evaluate_batch() are thread-safe; the
// underlying ONNX Runtime session is safe for concurrent inference.

#pragma once

#include "nn/backend.hpp"

#include <memory>
#include <string>
#include <vector>

// Forward-declare ORT types so callers don't need ONNX Runtime headers.
namespace Ort {
    struct Env;
    struct Session;
    struct MemoryInfo;
}  // namespace Ort

namespace chess_mlx::nn {

class Lc0Backend : public NNBackend {
public:
    // Construct from an ONNX file path.  Throws std::runtime_error if the
    // model cannot be loaded.
    explicit Lc0Backend(const std::string& onnx_path,
                        int intra_threads = 1,
                        int inter_threads = 1);

    ~Lc0Backend() override;

    // Disallow copy (owns non-copyable ORT session).
    Lc0Backend(const Lc0Backend&)            = delete;
    Lc0Backend& operator=(const Lc0Backend&) = delete;

    // ---------------------------------------------------------------------------
    // NNBackend interface
    // ---------------------------------------------------------------------------

    // Input: flat float32 array of size 64 * 112 = 7168 (LC0's [64, 112] layout).
    // Output policy: size 1858 (LC0's compact policy space).
    // MCTS must use Lc0ChessTraits (not ChessTraits) to interpret these indices.
    NNOutput evaluate(const std::vector<float>& input_tensor) override;

    std::vector<NNOutput> evaluate_batch(
        const std::vector<std::vector<float>>& inputs) override;

    std::string   name()        const override { return "lc0-onnx"; }
    std::size_t   input_size()  const override { return 64 * 112; }  // 7168
    std::size_t   policy_size() const override { return 1858; }

    // LC0 runs on CPU; pipelining still helps because ONNX Runtime's CPU
    // runtime can use multiple threads internally for a single batch.
    bool batch_preferred() const override { return true; }

    // Path to the loaded ONNX model.
    const std::string& onnx_path() const { return onnx_path_; }

private:
    // Run a single forward pass on a [B, 112, 8, 8] batch.
    // Returns raw policy [B, 1858], wdl [B, 3], mlh [B, 1].
    struct RawOutput {
        std::vector<float> policy;  // [B * 1858]
        std::vector<float> wdl;     // [B * 3]
        std::vector<float> mlh;     // [B * 1]
    };
    RawOutput run_ort(const float* input, std::size_t batch) const;

    // Convert our [64, 112] row-major layout → ONNX's [112, 8, 8].
    // LC0's ONNX input expects [B, 112, 8, 8] (channel-first).
    // Our encoder outputs [64, 112] where out[sq * 112 + plane].
    // Conversion: out_ort[plane * 64 + sq] = our[sq * 112 + plane].
    void transpose_to_ort(const float* src, float* dst, std::size_t batch) const;

    // Convert raw WDL logits [3] → softmax, return value = P(win) - P(loss).
    static float wdl_to_value(const float* wdl_logits);

    // ---
    std::string  onnx_path_;
    // ORT objects — owned via unique_ptr to hide implementation details.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace chess_mlx::nn
