// SPDX-License-Identifier: MIT
// engine/src/nn/lc0_backend.cpp
//
// Implementation of Lc0Backend using ONNX Runtime (CPU execution provider).
//
// Design notes:
//   * We use the C++ ONNX Runtime API (onnxruntime_cxx_api.h).
//   * Batch inference: the ONNX model accepts a dynamic batch dimension.
//   * The LC0 ONNX export expects input [B, 112, 8, 8] (channel-first).
//     Our encoder outputs [64, 112] per position (square-major), so we
//     must transpose: out[plane * 64 + sq] = src[sq * 112 + plane].
//   * WDL output is raw logits; we apply softmax and compute value = W - L.
//   * Policy output [B, 1858] is mapped to [B, 4672] by the precomputed table.

#include "nn/lc0_backend.hpp"

#ifdef CHESS_MLX_HAS_ONNX

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace chess_mlx::nn {

// Returns a per-host cache directory for CoreML compiled models.  We avoid
// per-process temp dirs because the CoreML EP recompiles on every cache miss,
// which can add many seconds to engine startup on large LC0 nets like BT4.
//
// Layout: $HOME/.cache/chess_mlx/coreml/<basename-of-onnx>/
// The CoreML EP itself sub-keys by model hash, so multiple sessions for the
// same ONNX share the cache and a model update invalidates the entries
// without needing manual cache eviction.
static std::string coreml_cache_dir(const std::string& onnx_path) {
    namespace fs = std::filesystem;
    const char* home = std::getenv("HOME");
    fs::path base = home ? fs::path(home) / ".cache" / "chess_mlx" / "coreml"
                         : fs::temp_directory_path() / "chess_mlx_coreml";
    base /= fs::path(onnx_path).stem().string();  // e.g. "BT4"
    std::error_code ec;
    fs::create_directories(base, ec);  // best-effort
    return base.string();
}

// Whether to attempt CoreML EP registration.  Driven by the
// CHESS_MLX_LC0_EP env var; default is OFF.
//
// Why off by default: for the BT4 net (740 MB, 15 transformer layers,
// dynamic batch dim) the CoreML EP on macOS 15+ partitions the graph
// into ~70 dynamic_mlprogram subgraphs and persists ~3.5 GB of
// compiled artefacts to disk.  Cold compile takes ~10 minutes; even
// warm load (cache hit) is in the 5-minute range because Apple's
// MLProgram runtime re-validates each subgraph per session.  Until we
// either:
//   (a) freeze the batch dim before passing the model to ORT (so
//       CoreML lands the whole graph on a single static MLProgram), or
//   (b) ship a much smaller LC0 net (e.g. t1_256_distilled) where the
//       partitioning overhead is amortised more favourably,
// CoreML on BT4 is a worse user experience than the CPU EP.
//
// For users who explicitly want to evaluate it (and accept the cold
// compile), set:
//   export CHESS_MLX_LC0_EP=coreml
//
// To silence the "running on CPU EP" info-string log:
//   export CHESS_MLX_LC0_EP=cpu
static bool env_enable_coreml() {
    const char* ep = std::getenv("CHESS_MLX_LC0_EP");
    if (!ep || !*ep) return false;           // default = CPU only
    const std::string s = ep;
    if (s == "cpu" || s == "off" || s == "0" || s == "false") return false;
    return s.find("coreml") != std::string::npos;
}

// ---------------------------------------------------------------------------
// Impl struct: holds ORT objects.
// ---------------------------------------------------------------------------
struct Lc0Backend::Impl {
    Ort::Env         env;
    Ort::Session     session;
    Ort::MemoryInfo  memory_info;

    // Input / output tensor names (owned C-strings from ORT).
    std::string input_name;
    std::string policy_name;
    std::string wdl_name;
    std::string mlh_name;

    Impl(const std::string& onnx_path, int intra_threads, int inter_threads)
        : env(ORT_LOGGING_LEVEL_WARNING, "lc0_backend"),
          session(nullptr),
          memory_info(Ort::MemoryInfo::CreateCpu(
              OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(intra_threads);
        opts.SetInterOpNumThreads(inter_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // -----------------------------------------------------------------
        // CoreML execution provider (Apple Silicon ANE / GPU).
        //
        // ORT-CPU on an M3 manages roughly 1-2 inferences/sec for a network
        // the size of BT4 (740 MB).  CoreML 5 + MLProgram running on ANE/GPU
        // is typically 10-50× faster per batch.  The first-run cost is a
        // 10-60 second model-compile; we persist the compiled artefact to a
        // hashed cache directory under $HOME/.cache so subsequent runs start
        // in well under a second.
        //
        // We append CoreML BEFORE the implicit CPU EP, so CoreML claims
        // every op it can run; whatever it refuses falls back to the CPU EP
        // automatically.  If the registration itself fails (e.g. no Apple
        // Silicon, ORT built without CoreML support), the catch below logs
        // it and we keep going on CPU.
        // -----------------------------------------------------------------
        const bool coreml_requested = env_enable_coreml();
        bool coreml_active = false;
        if (coreml_requested) {
            try {
                const std::string cache_dir = coreml_cache_dir(onnx_path);
                std::unordered_map<std::string, std::string> coreml_opts{
                    // MLProgram is CoreML 5+; supports a wider op set and runs
                    // more reliably on the Neural Engine than the older
                    // NeuralNetwork format.
                    {"ModelFormat",            "MLProgram"},
                    // Let CoreML pick the best engine per op (ANE for what fits,
                    // Metal GPU for the rest).
                    {"MLComputeUnits",         "ALL"},
                    // BT4 input is [B, 112, 8, 8] with dynamic batch — allow
                    // dynamic-shape ops so the whole graph lands on CoreML
                    // instead of partitioning around the batch dim.
                    {"RequireStaticInputShapes", "0"},
                    {"EnableOnSubgraphs",       "0"},
                    // Persist compiled artefacts; key derived per model dir.
                    {"ModelCacheDirectory",    cache_dir},
                };
                opts.AppendExecutionProvider("CoreML", coreml_opts);
                coreml_active = true;
                std::cerr << "info string Lc0Backend: CoreML EP appended, cache="
                          << cache_dir << "\n";
            } catch (const std::exception& e) {
                std::cerr << "info string Lc0Backend: CoreML EP unavailable ("
                          << e.what() << "); using CPU EP\n";
            }
        }
        if (!coreml_active) {
            std::cerr << "info string Lc0Backend: running on CPU EP "
                      << "(set CHESS_MLX_LC0_EP=coreml to opt in, "
                      << "or =cpu to silence this)\n";
        }

        session = Ort::Session(env, onnx_path.c_str(), opts);

        // Discover I/O names
        Ort::AllocatorWithDefaultOptions alloc;

        auto in_name_ptr = session.GetInputNameAllocated(0, alloc);
        input_name = in_name_ptr.get();

        // Scan outputs to find policy / wdl / mlh by suffix
        const std::size_t n_out = session.GetOutputCount();
        for (std::size_t i = 0; i < n_out; ++i) {
            auto name_ptr = session.GetOutputNameAllocated(i, alloc);
            std::string name(name_ptr.get());
            if (name.rfind("policy") != std::string::npos)      policy_name = name;
            else if (name.rfind("wdl") != std::string::npos)    wdl_name    = name;
            else if (name.rfind("mlh") != std::string::npos)    mlh_name    = name;
        }
        if (policy_name.empty() || wdl_name.empty() || mlh_name.empty()) {
            throw std::runtime_error(
                "Lc0Backend: could not find policy/wdl/mlh outputs in " + onnx_path);
        }
    }
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
Lc0Backend::Lc0Backend(const std::string& onnx_path,
                        int intra_threads,
                        int inter_threads)
    : onnx_path_(onnx_path),
      impl_(std::make_unique<Impl>(onnx_path, intra_threads, inter_threads))
{
    std::cerr << "info string Lc0Backend loaded: " << onnx_path << "\n";
}

Lc0Backend::~Lc0Backend() = default;

// ---------------------------------------------------------------------------
// Transpose helper: [B, 64, 112] → [B, 112, 64] (equiv [B, 112, 8, 8])
// Our layout:  src[b * 7168 + sq * 112 + plane]
// ORT layout:  dst[b * 7168 + plane * 64 + sq]
// ---------------------------------------------------------------------------
void Lc0Backend::transpose_to_ort(const float* src, float* dst,
                                   std::size_t batch) const
{
    constexpr int kSq    = 64;
    constexpr int kPlane = 112;
    constexpr int kStride = kSq * kPlane;  // 7168

    for (std::size_t b = 0; b < batch; ++b) {
        const float* s = src + b * static_cast<std::size_t>(kStride);
        float*       d = dst + b * static_cast<std::size_t>(kStride);
        for (int plane = 0; plane < kPlane; ++plane) {
            for (int sq = 0; sq < kSq; ++sq) {
                d[plane * kSq + sq] = s[sq * kPlane + plane];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WDL softmax → scalar value = P(win) - P(loss).
// ---------------------------------------------------------------------------
float Lc0Backend::wdl_to_value(const float* wdl_logits)
{
    float w = wdl_logits[0];
    float d = wdl_logits[1];
    float l = wdl_logits[2];
    // Numerically stable softmax
    const float mx = std::max({w, d, l});
    w = std::exp(w - mx);
    d = std::exp(d - mx);
    l = std::exp(l - mx);
    const float sum = w + d + l;
    // value = P(win) - P(loss) in [-1, +1]
    return (w - l) / sum;
}

// ---------------------------------------------------------------------------
// Run ORT inference.
// ---------------------------------------------------------------------------
Lc0Backend::RawOutput Lc0Backend::run_ort(const float* input,
                                           std::size_t batch) const
{
    constexpr std::size_t kInputStride = 64 * 112;

    // Transpose to channel-first [B, 112, 8, 8]
    std::vector<float> ort_input(batch * kInputStride);
    transpose_to_ort(input, ort_input.data(), batch);

    // Build input tensor
    const std::array<std::int64_t, 4> input_shape{
        static_cast<std::int64_t>(batch), 112, 8, 8
    };
    auto input_tensor = Ort::Value::CreateTensor<float>(
        impl_->memory_info,
        ort_input.data(),
        ort_input.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Output names
    const char* const output_names[3] = {
        impl_->policy_name.c_str(),
        impl_->wdl_name.c_str(),
        impl_->mlh_name.c_str()
    };
    const char* const input_names[1] = { impl_->input_name.c_str() };

    auto outputs = impl_->session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 3
    );

    // Extract raw data
    RawOutput raw;
    raw.policy.resize(batch * 1858);
    raw.wdl.resize(batch * 3);
    raw.mlh.resize(batch * 1);

    std::memcpy(raw.policy.data(),
                outputs[0].GetTensorData<float>(),
                batch * 1858 * sizeof(float));
    std::memcpy(raw.wdl.data(),
                outputs[1].GetTensorData<float>(),
                batch * 3 * sizeof(float));
    std::memcpy(raw.mlh.data(),
                outputs[2].GetTensorData<float>(),
                batch * 1 * sizeof(float));
    return raw;
}

// ---------------------------------------------------------------------------
// evaluate() — single position.
// ---------------------------------------------------------------------------
NNOutput Lc0Backend::evaluate(const std::vector<float>& input_tensor)
{
    const auto raw = run_ort(input_tensor.data(), 1);

    NNOutput out;
    // Output policy in LC0's 1858-slot space.
    // MCTS must use Lc0ChessTraits so that move_to_policy_idx returns LC0 slots.
    out.policy.assign(raw.policy.begin(), raw.policy.end());  // 1858 floats
    out.value      = wdl_to_value(raw.wdl.data());
    out.moves_left = raw.mlh[0];
    return out;
}

// ---------------------------------------------------------------------------
// evaluate_batch() — N positions.
// ---------------------------------------------------------------------------
std::vector<NNOutput> Lc0Backend::evaluate_batch(
    const std::vector<std::vector<float>>& inputs)
{
    if (inputs.empty()) return {};

    const std::size_t B = inputs.size();
    constexpr std::size_t kStride = 64 * 112;

    // Flatten inputs into one contiguous buffer
    std::vector<float> flat(B * kStride);
    for (std::size_t i = 0; i < B; ++i) {
        std::memcpy(flat.data() + i * kStride,
                    inputs[i].data(),
                    kStride * sizeof(float));
    }

    auto raw = run_ort(flat.data(), B);

    // Share one batched logit buffer across all NNOutput results.
    // raw.policy is already a contiguous [B * 1858] vector; move it into a
    // shared_ptr so leaves can read in-place via NNOutput::policy_at().
    auto shared_pol = std::make_shared<std::vector<float>>(std::move(raw.policy));

    std::vector<NNOutput> results(B);
    for (std::size_t i = 0; i < B; ++i) {
        results[i].policy_shared = shared_pol;
        results[i].policy_offset = i * 1858;
        results[i].policy_extent = 1858;
        results[i].value      = wdl_to_value(raw.wdl.data() + i * 3);
        results[i].moves_left = raw.mlh[i];
    }
    return results;
}

}  // namespace chess_mlx::nn

#else  // CHESS_MLX_HAS_ONNX not defined

// Stub so the file compiles when ONNX Runtime is not available.
#include <stdexcept>

namespace chess_mlx::nn {

Lc0Backend::Lc0Backend(const std::string& path, int, int)
    : onnx_path_(path)
{
    throw std::runtime_error("Lc0Backend: ONNX Runtime not compiled in");
}

Lc0Backend::~Lc0Backend() = default;

NNOutput Lc0Backend::evaluate(const std::vector<float>&) {
    throw std::runtime_error("Lc0Backend: ONNX Runtime not compiled in");
}

std::vector<NNOutput> Lc0Backend::evaluate_batch(
    const std::vector<std::vector<float>>&) {
    throw std::runtime_error("Lc0Backend: ONNX Runtime not compiled in");
}

}  // namespace chess_mlx::nn

#endif  // CHESS_MLX_HAS_ONNX
