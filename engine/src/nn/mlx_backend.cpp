// SPDX-License-Identifier: MIT
// engine/src/nn/mlx_backend.cpp
//
// MlxBackend implementation.  Loads a ChessShogiTransformer checkpoint from
// a safetensors file via MLX-C and runs forward passes on the Apple GPU.
//
// This implementation follows the architecture in
// `training/src/training/models/transformer.py`:
//   * Per-game input projection (Linear) + positional embedding
//   * N transformer layers, each of which is:
//       Attention(x)  := LayerNorm(x + α * out_proj(Smolgen-biased MHA(x)))
//       FFN(x)        := LayerNorm(x + α * fc2(Mish(fc1(x))))
//   * Final LayerNorm
//   * Three heads: policy, value, moves-left (all mean-pool → MLP)
//
// The weights in the safetensors file follow MLX's `save_weights` convention:
//   * `chess_input_proj.weight` / `.bias` (shape [d_model, feat_dim] / [d_model])
//   * `chess_pos_embed.weight`   (shape [seq_len, d_model])
//   * `chess_layers.<i>.attn.q_proj.weight`       (shape [d_model, d_model])
//   * `chess_layers.<i>.attn.k_proj.weight`       (shape [d_model, d_model])
//   * `chess_layers.<i>.attn.v_proj.weight`       (shape [d_model, d_model])
//   * `chess_layers.<i>.attn.out_proj.weight/.bias`
//   * `chess_layers.<i>.attn.norm.weight/.bias`
//   * `chess_layers.<i>.attn.smolgen.compress.weight/.bias`
//   * `chess_layers.<i>.attn.smolgen.fc1.weight/.bias`
//   * `chess_layers.<i>.attn.smolgen.fc2.weight/.bias`
//   * `chess_layers.<i>.ffn.fc1.weight/.bias`
//   * `chess_layers.<i>.ffn.fc2.weight/.bias`
//   * `chess_layers.<i>.ffn.norm.weight/.bias`
//   * `final_norm.weight/.bias`
//   * `chess_policy_head.fc1.weight/.bias`  (or shogi_policy_head)
//   * `chess_policy_head.fc2.weight/.bias`
//   * `chess_policy_head.norm.weight/.bias`
//   * `value_head.fc1.weight/.bias`, `value_head.fc2.weight/.bias`, `value_head.norm.weight/.bias`
//   * `moves_left_head.fc1.weight/.bias`, `moves_left_head.fc2.weight/.bias`, `moves_left_head.norm.weight/.bias`
//
// Notable pitfalls:
//   * `nn.Linear(in, out)` stores weight as shape [out, in], so `y = x @ W.T + b`.
//     MLX-C has `mlx_addmm(c, a, b, alpha, beta)` → alpha*a@b + beta*c.
//   * LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//     No direct op — implemented manually.
//
// Error-handling convention: every helper checks the return code from MLX and
// throws std::runtime_error on failure.  Constructor failures propagate back
// to the backend factory, which falls back to StubBackend.

#include "nn/mlx_backend.hpp"

#ifdef CHESS_MLX_HAS_MLX

#include <mlx/c/mlx.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace chess_mlx::nn {

namespace {

// MLX's default error handler calls exit(-1) which is lethal to unit tests
// and to any higher-level error recovery.  Replace it with one that merely
// logs to stderr.  Installed once, from within MlxBackend's constructor.
static void mlx_quiet_error_handler(const char* msg, void* /*data*/) {
    std::cerr << "info string MLX error: " << (msg ? msg : "(null)") << "\n";
}

static std::once_flag g_mlx_handler_flag;
static void install_error_handler() {
    std::call_once(g_mlx_handler_flag, []() {
        mlx_set_error_handler(&mlx_quiet_error_handler, nullptr, nullptr);
    });
}

// -----------------------------------------------------------------------------
// Tiny JSON extractor — just enough to read our sidecar.  The sidecar is a
// fixed format produced by training/src/training/export.py; anything unusual
// results in a fallback to defaults (with a warning).
// -----------------------------------------------------------------------------
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static bool json_extract_int(const std::string& s, const std::string& key, int& out) {
    const std::string needle = "\"" + key + "\"";
    auto p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (p >= s.size()) return false;
    char* end = nullptr;
    long val = std::strtol(s.c_str() + p, &end, 10);
    if (end == s.c_str() + p) return false;
    out = static_cast<int>(val);
    return true;
}

static bool json_extract_str(const std::string& s, const std::string& key, std::string& out) {
    const std::string needle = "\"" + key + "\"";
    auto p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    p = s.find('"', p);
    if (p == std::string::npos) return false;
    const auto end = s.find('"', p + 1);
    if (end == std::string::npos) return false;
    out = s.substr(p + 1, end - p - 1);
    return true;
}

static MlxModelConfig parse_sidecar(const std::string& weights_path) {
    MlxModelConfig cfg;
    // Replace .safetensors with .json
    std::string json_path = weights_path;
    const auto pos = json_path.rfind(".safetensors");
    if (pos != std::string::npos) json_path.replace(pos, 12, ".json");
    else                          json_path += ".json";

    try {
        const auto txt = read_file(json_path);
        json_extract_str(txt, "game",     cfg.game);
        json_extract_int(txt, "n_layers", cfg.n_layers);
        json_extract_int(txt, "d_model",  cfg.d_model);
        json_extract_int(txt, "n_heads",  cfg.n_heads);
        json_extract_int(txt, "ffn_dim",  cfg.ffn_dim);
        json_extract_int(txt, "seq_len",  cfg.seq_len);
        json_extract_int(txt, "feat_dim", cfg.feat_dim);
        json_extract_int(txt, "num_moves", cfg.num_moves);
    } catch (const std::exception& e) {
        std::cerr << "info string MlxBackend: no sidecar at " << json_path
                  << " (" << e.what() << "); using defaults\n";
    }
    return cfg;
}

// -----------------------------------------------------------------------------
// Small RAII helper for mlx_array so we don't leak on exceptions.
// -----------------------------------------------------------------------------
struct MlxArr {
    mlx_array arr{};
    bool      owned{true};
    MlxArr() : arr(mlx_array_new()), owned(true) {}
    explicit MlxArr(mlx_array a, bool own = true) : arr(a), owned(own) {}
    MlxArr(const MlxArr&) = delete;
    MlxArr& operator=(const MlxArr&) = delete;
    MlxArr(MlxArr&& o) noexcept : arr(o.arr), owned(o.owned) { o.owned = false; }
    MlxArr& operator=(MlxArr&& o) noexcept {
        if (this != &o) {
            if (owned) mlx_array_free(arr);
            arr   = o.arr;
            owned = o.owned;
            o.owned = false;
        }
        return *this;
    }
    ~MlxArr() { if (owned) mlx_array_free(arr); }
    mlx_array* ptr() { return &arr; }
    operator mlx_array() const { return arr; }
};

// -----------------------------------------------------------------------------
// Linear layer: y = x @ W.T + b  (W shape = [out, in], b shape = [out]).
// -----------------------------------------------------------------------------
static MlxArr linear(const mlx_array& x,
                     const mlx_array& w,
                     const mlx_array* b,
                     mlx_stream s) {
    // Matmul expects plain shapes; x can be [*, in].  W is [out, in].
    // We compute x @ W.T.  MLX has no transpose on matmul; we transpose W.
    MlxArr w_t;
    int axes[] = {1, 0};
    if (mlx_transpose_axes(w_t.ptr(), w, axes, 2, s) != 0)
        throw std::runtime_error("linear: transpose failed");

    MlxArr out;
    if (mlx_matmul(out.ptr(), x, w_t.arr, s) != 0)
        throw std::runtime_error("linear: matmul failed");

    if (b != nullptr) {
        MlxArr biased;
        if (mlx_add(biased.ptr(), out.arr, *b, s) != 0)
            throw std::runtime_error("linear: add bias failed");
        return biased;
    }
    return out;
}

// -----------------------------------------------------------------------------
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta  (axis = -1)
// Uses mlx_fast_layer_norm which is a single fused Metal kernel (vs 8+ ops).
// -----------------------------------------------------------------------------
static MlxArr layer_norm(const mlx_array& x,
                         const mlx_array& gamma,
                         const mlx_array& beta,
                         float eps,
                         mlx_stream s) {
    MlxArr out;
    if (mlx_fast_layer_norm(out.ptr(), x, gamma, beta, eps, s) != 0) {
        // Fallback to manual implementation if fast path fails.
        const int ndim = static_cast<int>(mlx_array_ndim(x));
        const int last_axis = ndim - 1;
        MlxArr mean;
        mlx_mean_axis(mean.ptr(), x, last_axis, true, s);
        MlxArr centered;
        mlx_subtract(centered.ptr(), x, mean.arr, s);
        MlxArr squared;
        mlx_multiply(squared.ptr(), centered.arr, centered.arr, s);
        MlxArr var;
        mlx_mean_axis(var.ptr(), squared.arr, last_axis, true, s);
        MlxArr eps_arr(mlx_array_new_float32(eps));
        MlxArr var_eps;
        mlx_add(var_eps.ptr(), var.arr, eps_arr.arr, s);
        MlxArr std_dev;
        mlx_sqrt(std_dev.ptr(), var_eps.arr, s);
        MlxArr normed;
        mlx_divide(normed.ptr(), centered.arr, std_dev.arr, s);
        MlxArr scaled;
        mlx_multiply(scaled.ptr(), normed.arr, gamma, s);
        MlxArr shifted;
        mlx_add(shifted.ptr(), scaled.arr, beta, s);
        return shifted;
    }
    return out;
}

// -----------------------------------------------------------------------------
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715 * x^3)))
// Use the simpler form:  GELU(x) = x * sigmoid(1.702 * x) which is close
// enough for inference.  But we'll do the exact tanh-form for fidelity.
// -----------------------------------------------------------------------------
static MlxArr gelu(const mlx_array& x, mlx_stream s) {
    const float kBeta = 0.7978845608028654f;    // sqrt(2/pi)
    const float kKappa = 0.044715f;

    MlxArr three(mlx_array_new_float32(3.0f));
    MlxArr x_cubed;
    if (mlx_power(x_cubed.ptr(), x, three.arr, s) != 0)
        throw std::runtime_error("gelu: power failed");

    MlxArr kappa(mlx_array_new_float32(kKappa));
    MlxArr kx3;
    if (mlx_multiply(kx3.ptr(), x_cubed.arr, kappa.arr, s) != 0)
        throw std::runtime_error("gelu: kx3 failed");

    MlxArr inner;
    if (mlx_add(inner.ptr(), x, kx3.arr, s) != 0)
        throw std::runtime_error("gelu: inner add failed");

    MlxArr beta(mlx_array_new_float32(kBeta));
    MlxArr scaled;
    if (mlx_multiply(scaled.ptr(), inner.arr, beta.arr, s) != 0)
        throw std::runtime_error("gelu: scaled failed");

    MlxArr t;
    if (mlx_tanh(t.ptr(), scaled.arr, s) != 0)
        throw std::runtime_error("gelu: tanh failed");

    MlxArr one(mlx_array_new_float32(1.0f));
    MlxArr one_plus_t;
    if (mlx_add(one_plus_t.ptr(), one.arr, t.arr, s) != 0)
        throw std::runtime_error("gelu: one+tanh failed");

    MlxArr half(mlx_array_new_float32(0.5f));
    MlxArr half_x;
    if (mlx_multiply(half_x.ptr(), half.arr, x, s) != 0)
        throw std::runtime_error("gelu: half*x failed");

    MlxArr out;
    if (mlx_multiply(out.ptr(), half_x.arr, one_plus_t.arr, s) != 0)
        throw std::runtime_error("gelu: final multiply failed");

    return out;
}

// Softplus(x) = log(1 + exp(x))
// More numerically stable:  softplus(x) = max(x, 0) + log1p(exp(-|x|))
// For inference with bounded inputs, the naive form is fine.
static MlxArr softplus(const mlx_array& x, mlx_stream s) {
    MlxArr ex;
    if (mlx_exp(ex.ptr(), x, s) != 0)
        throw std::runtime_error("softplus: exp failed");
    MlxArr one(mlx_array_new_float32(1.0f));
    MlxArr one_plus_e;
    if (mlx_add(one_plus_e.ptr(), one.arr, ex.arr, s) != 0)
        throw std::runtime_error("softplus: add failed");
    MlxArr out;
    if (mlx_log(out.ptr(), one_plus_e.arr, s) != 0)
        throw std::runtime_error("softplus: log failed");
    return out;
}

// Mish(x) = x * tanh(softplus(x))
static MlxArr mish(const mlx_array& x, mlx_stream s) {
    auto sp = softplus(x, s);
    MlxArr t;
    if (mlx_tanh(t.ptr(), sp.arr, s) != 0)
        throw std::runtime_error("mish: tanh failed");
    MlxArr out;
    if (mlx_multiply(out.ptr(), x, t.arr, s) != 0)
        throw std::runtime_error("mish: multiply failed");
    return out;
}

// Reshape helper — makes error propagation cleaner.
static MlxArr reshape(const mlx_array& x, const std::vector<int>& shape, mlx_stream s) {
    MlxArr out;
    if (mlx_reshape(out.ptr(), x, shape.data(), shape.size(), s) != 0)
        throw std::runtime_error("reshape failed");
    return out;
}
static MlxArr transpose_axes(const mlx_array& x, const std::vector<int>& axes, mlx_stream s) {
    MlxArr out;
    if (mlx_transpose_axes(out.ptr(), x, axes.data(), axes.size(), s) != 0)
        throw std::runtime_error("transpose failed");
    return out;
}
static MlxArr matmul(const mlx_array& a, const mlx_array& b, mlx_stream s) {
    MlxArr out;
    if (mlx_matmul(out.ptr(), a, b, s) != 0)
        throw std::runtime_error("matmul failed");
    return out;
}
static MlxArr add(const mlx_array& a, const mlx_array& b, mlx_stream s) {
    MlxArr out;
    if (mlx_add(out.ptr(), a, b, s) != 0)
        throw std::runtime_error("add failed");
    return out;
}
static MlxArr multiply(const mlx_array& a, const mlx_array& b, mlx_stream s) {
    MlxArr out;
    if (mlx_multiply(out.ptr(), a, b, s) != 0)
        throw std::runtime_error("multiply failed");
    return out;
}
static MlxArr divide(const mlx_array& a, const mlx_array& b, mlx_stream s) {
    MlxArr out;
    if (mlx_divide(out.ptr(), a, b, s) != 0)
        throw std::runtime_error("divide failed");
    return out;
}
static MlxArr softmax(const mlx_array& x, int axis, mlx_stream s) {
    MlxArr out;
    if (mlx_softmax_axis(out.ptr(), x, axis, false, s) != 0)
        throw std::runtime_error("softmax failed");
    return out;
}
static MlxArr mean_axis(const mlx_array& x, int axis, bool keepdims, mlx_stream s) {
    MlxArr out;
    if (mlx_mean_axis(out.ptr(), x, axis, keepdims, s) != 0)
        throw std::runtime_error("mean failed");
    return out;
}
static MlxArr tanh_arr(const mlx_array& x, mlx_stream s) {
    MlxArr out;
    if (mlx_tanh(out.ptr(), x, s) != 0)
        throw std::runtime_error("tanh failed");
    return out;
}
static MlxArr broadcast_to(const mlx_array& x, const std::vector<int>& shape, mlx_stream s) {
    MlxArr out;
    if (mlx_broadcast_to(out.ptr(), x, shape.data(), shape.size(), s) != 0)
        throw std::runtime_error("broadcast_to failed");
    return out;
}

} // namespace

// -----------------------------------------------------------------------------
// MlxBackend::Impl
// -----------------------------------------------------------------------------
struct MlxBackend::Impl {
    // `load_stream` is used only for `mlx_load_safetensors` because the
    // Load op has no GPU implementation (Load::eval_gpu is unimplemented
    // in MLX 0.31). After the initial load, every inference op runs on
    // `stream`, which is Metal/GPU when MLX was built with Metal, and
    // falls back to CPU otherwise. MLX automatically migrates array data
    // between devices on first use.
    mlx_stream stream;
    mlx_stream load_stream;
    // Owned arrays indexed by parameter name (as saved by MLX `save_weights`).
    std::unordered_map<std::string, mlx_array> params;

    std::string prefix;  // "chess_" or "shogi_"

    // Compiled forward-pass closure produced by mlx_compile(shapeless=true).
    // Once set, forward_batch_nolock dispatches through this instead of
    // rebuilding the computation graph on every call.
    mlx_closure compiled_closure{nullptr};
    bool has_compiled_closure{false};

    // Config needed inside the closure function.
    int n_layers{0}, d_model{0}, n_heads{0}, seq_len{0}, feat_dim{0};

    Impl() {
        load_stream = mlx_default_cpu_stream_new();
        stream = mlx_default_gpu_stream_new();
        compiled_closure = mlx_closure_new();
    }
    ~Impl() {
        for (auto& kv : params) mlx_array_free(kv.second);
        params.clear();
        mlx_closure_free(compiled_closure);
        mlx_stream_free(stream);
        mlx_stream_free(load_stream);
    }

    const mlx_array& get(const std::string& name) const {
        auto it = params.find(name);
        if (it == params.end())
            throw std::runtime_error("MlxBackend: missing parameter '" + name + "'");
        return it->second;
    }

    bool has(const std::string& name) const {
        return params.find(name) != params.end();
    }

    // ---------------------------------------------------------------------------
    // Core forward-graph builder.
    //
    // Receives the batch tensor `x_in` [B, S, F] and populates `outputs` with
    // {policy_logits, value, moves_left}.  Does NOT call mlx_array_eval — the
    // caller materialises the outputs.  Returns 0 on success.
    //
    // This method is the single authoritative graph-building path.  The static
    // mlx_closure callback below delegates to it so that mlx_compile can wrap
    // and cache the compiled Metal shaders.
    // ---------------------------------------------------------------------------
    int build_graph(mlx_vector_array* outputs, mlx_array x_in) const {
        mlx_stream s = stream;
        const std::string& pfx = prefix;

        const int D  = d_model;
        const int H  = n_heads;
        const int HD = D / H;
        const int S_  = seq_len;
        const float eps = 1e-5f;

        // B is dynamic — read from shape so shapeless compile can handle any batch.
        const int B = mlx_array_dim(x_in, 0);

        MlxArr h;
        {
            MlxArr x_holder(x_in);  // RAII ownership

            const std::string in_proj_w = pfx + "input_proj.weight";
            const std::string in_proj_b = pfx + "input_proj.bias";
            h = linear(x_holder.arr,
                       get(in_proj_w),
                       has(in_proj_b) ? &get(in_proj_b) : nullptr,
                       s);

            const std::string pos_emb = pfx + "pos_embed.weight";
            MlxArr pe = reshape(get(pos_emb), {1, S_, D}, s);
            h = add(h.arr, pe.arr, s);
        }

        const float scale_val = std::sqrt(static_cast<float>(HD));
        MlxArr inv_scale(mlx_array_new_float32(1.0f / scale_val));

        for (int layer = 0; layer < n_layers; ++layer) {
            const std::string base = pfx + "layers." + std::to_string(layer) + ".";

            // --- Smolgen bias ---
            MlxArr compressed_pre = linear(h.arr,
                                    get(base + "attn.smolgen.compress.weight"),
                                    has(base + "attn.smolgen.compress.bias") ? &get(base + "attn.smolgen.compress.bias") : nullptr,
                                    s);
            MlxArr compressed = gelu(compressed_pre.arr, s);
            MlxArr sm_pooled = mean_axis(compressed.arr, 1, false, s);
            MlxArr gen_pre = linear(sm_pooled.arr,
                                    get(base + "attn.smolgen.fc1.weight"),
                                    has(base + "attn.smolgen.fc1.bias") ? &get(base + "attn.smolgen.fc1.bias") : nullptr,
                                    s);
            MlxArr gen = gelu(gen_pre.arr, s);
            MlxArr bias_flat_arr = linear(gen.arr,
                                    get(base + "attn.smolgen.fc2.weight"),
                                    has(base + "attn.smolgen.fc2.bias") ? &get(base + "attn.smolgen.fc2.bias") : nullptr,
                                    s);
            MlxArr bias_mat = reshape(bias_flat_arr.arr, {B, 1, S_, S_}, s);
            MlxArr smolgen_bias = broadcast_to(bias_mat.arr, {B, H, S_, S_}, s);

            // --- Q/K/V projections ---
            MlxArr q = linear(h.arr, get(base + "attn.q_proj.weight"), nullptr, s);
            MlxArr k = linear(h.arr, get(base + "attn.k_proj.weight"), nullptr, s);
            MlxArr v = linear(h.arr, get(base + "attn.v_proj.weight"), nullptr, s);

            q = reshape(q.arr, {B, S_, H, HD}, s);
            q = transpose_axes(q.arr, {0, 2, 1, 3}, s);
            k = reshape(k.arr, {B, S_, H, HD}, s);
            k = transpose_axes(k.arr, {0, 2, 1, 3}, s);
            v = reshape(v.arr, {B, S_, H, HD}, s);
            v = transpose_axes(v.arr, {0, 2, 1, 3}, s);

            MlxArr attn_raw;
            const float sdpa_scale = 1.0f / scale_val;
            mlx_array null_sinks = mlx_array_new();
            if (mlx_fast_scaled_dot_product_attention(
                    attn_raw.ptr(), q.arr, k.arr, v.arr,
                    sdpa_scale, "array", smolgen_bias.arr, null_sinks, s) != 0) {
                MlxArr k_t = transpose_axes(k.arr, {0, 1, 3, 2}, s);
                MlxArr logits = matmul(q.arr, k_t.arr, s);
                MlxArr scaled_logits = multiply(logits.arr, inv_scale.arr, s);
                MlxArr biased_logits = add(scaled_logits.arr, smolgen_bias.arr, s);
                MlxArr attn_w = softmax(biased_logits.arr, -1, s);
                attn_raw = matmul(attn_w.arr, v.arr, s);
            }
            mlx_array_free(null_sinks);

            MlxArr attn_out = transpose_axes(attn_raw.arr, {0, 2, 1, 3}, s);
            attn_out = reshape(attn_out.arr, {B, S_, D}, s);

            const std::string ow  = base + "attn.out_proj.weight";
            const std::string ob  = base + "attn.out_proj.bias";
            const std::string an_w = base + "attn.norm.weight";
            const std::string an_b = base + "attn.norm.bias";

            MlxArr attn_proj = linear(attn_out.arr,
                                    get(ow),
                                    has(ob) ? &get(ob) : nullptr,
                                    s);
            MlxArr pre_norm = add(h.arr, attn_proj.arr, s);
            h = layer_norm(pre_norm.arr, get(an_w), get(an_b), eps, s);

            // --- FFN sub-layer ---
            const std::string f1w = base + "ffn.fc1.weight";
            const std::string f1b = base + "ffn.fc1.bias";
            const std::string f2w = base + "ffn.fc2.weight";
            const std::string f2b = base + "ffn.fc2.bias";
            const std::string fn_w = base + "ffn.norm.weight";
            const std::string fn_b = base + "ffn.norm.bias";

            MlxArr ffn_h1 = linear(h.arr,
                                   get(f1w),
                                   has(f1b) ? &get(f1b) : nullptr,
                                   s);
            MlxArr ffn_act = mish(ffn_h1.arr, s);
            MlxArr ffn_out = linear(ffn_act.arr,
                                    get(f2w),
                                    has(f2b) ? &get(f2b) : nullptr,
                                    s);
            MlxArr ffn_pre = add(h.arr, ffn_out.arr, s);
            h = layer_norm(ffn_pre.arr, get(fn_w), get(fn_b), eps, s);
        }

        h = layer_norm(h.arr, get("final_norm.weight"), get("final_norm.bias"), eps, s);

        auto run_head = [&](const std::string& head_pfx) -> MlxArr {
            MlxArr pooled = mean_axis(h.arr, 1, false, s);
            MlxArr normed = layer_norm(pooled.arr,
                                       get(head_pfx + ".norm.weight"),
                                       get(head_pfx + ".norm.bias"),
                                       eps, s);
            MlxArr h1 = linear(normed.arr,
                                get(head_pfx + ".fc1.weight"),
                                has(head_pfx + ".fc1.bias") ? &get(head_pfx + ".fc1.bias") : nullptr,
                                s);
            MlxArr act = gelu(h1.arr, s);
            return linear(act.arr,
                          get(head_pfx + ".fc2.weight"),
                          has(head_pfx + ".fc2.bias") ? &get(head_pfx + ".fc2.bias") : nullptr,
                          s);
        };

        MlxArr policy_logits  = run_head(pfx + "policy_head");
        MlxArr value_pre      = run_head("value_head");
        MlxArr moves_left_pre = run_head("moves_left_head");

        MlxArr value      = tanh_arr(value_pre.arr, s);
        MlxArr moves_left = softplus(moves_left_pre.arr, s);

        // Use mlx_vector_array_set_data to populate the output vector.
        // mlx_vector_array_append_value internally calls mlx_vector_array_get_
        // which throws on a null-ctx vector (e.g. mlx_vector_array_new_() used
        // inside mlx_closure_new_func_payload's lambda).  set_data handles
        // null-ctx by allocating the internal storage first.
        const mlx_array out_arrays[3] = {
            policy_logits.arr, value.arr, moves_left.arr
        };
        mlx_vector_array_set_data(outputs, out_arrays, 3);

        return 0;
    }

    // Static closure callback for mlx_closure_new_func_payload.
    // Signature: int(mlx_vector_array*, const mlx_vector_array, void*)
    // `payload` is a non-owning `Impl*`.
    static int closure_cb(mlx_vector_array* outputs,
                          const mlx_vector_array inputs,
                          void* payload) {
        auto* self = static_cast<MlxBackend::Impl*>(payload);
        mlx_array x_in = mlx_array_new();
        if (mlx_vector_array_get(&x_in, inputs, 0) != 0) {
            mlx_array_free(x_in);
            return -1;
        }
        return self->build_graph(outputs, x_in);
        // x_in ownership transferred into build_graph which wraps it in MlxArr RAII
    }
};

// -----------------------------------------------------------------------------
// Constructor / destructor
// -----------------------------------------------------------------------------

MlxBackend::MlxBackend(const std::string& weights_path,
                        std::size_t        engine_policy_size,
                        std::size_t        input_size)
    : engine_policy_size_(engine_policy_size), input_size_(input_size) {
    install_error_handler();
    config_ = parse_sidecar(weights_path);
    if (config_.game.empty()) config_.game = "chess";

    impl_ = std::make_unique<Impl>();
    impl_->prefix   = (config_.game == "shogi") ? "shogi_" : "chess_";
    impl_->n_layers = config_.n_layers;
    impl_->d_model  = config_.d_model;
    impl_->n_heads  = config_.n_heads;
    impl_->seq_len  = config_.seq_len;
    impl_->feat_dim = config_.feat_dim;

    mlx_map_string_to_array data     = mlx_map_string_to_array_new();
    mlx_map_string_to_string metadata = mlx_map_string_to_string_new();

    if (mlx_load_safetensors(&data, &metadata, weights_path.c_str(), impl_->load_stream) != 0) {
        mlx_map_string_to_array_free(data);
        mlx_map_string_to_string_free(metadata);
        throw std::runtime_error("MlxBackend: failed to load " + weights_path);
    }

    // Transfer ownership of every tensor into `impl_->params`.
    mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(data);
    const char* key = nullptr;
    mlx_array value = mlx_array_new();
    while (!mlx_map_string_to_array_iterator_next(&key, &value, it)) {
        mlx_array copy = mlx_array_new();
        // `value` is owned by the map — make a copy that we own.
        mlx_array_set(&copy, value);
        impl_->params.emplace(std::string(key), copy);
    }
    mlx_array_free(value);
    mlx_map_string_to_array_iterator_free(it);
    mlx_map_string_to_array_free(data);
    mlx_map_string_to_string_free(metadata);

    name_ = "mlx-" + config_.game + "-L" + std::to_string(config_.n_layers)
          + "-D" + std::to_string(config_.d_model);

    std::cerr << "info string MlxBackend: loaded " << impl_->params.size()
              << " tensors from " << weights_path
              << "; game=" << config_.game
              << " layers=" << config_.n_layers
              << " d_model=" << config_.d_model
              << " seq_len=" << config_.seq_len
              << " feat_dim=" << config_.feat_dim
              << " num_moves=" << config_.num_moves << "\n";

    // -------------------------------------------------------------------------
    // Sanity check: verify the sidecar's declared dims line up with the
    // weight tensors we just loaded.  Catches three failure modes:
    //
    //  1. Sidecar missing / malformed → defaults applied; if the model is
    //     actually shogi (or any non-default architecture) the first-layer
    //     input projection's shape won't match.
    //  2. Sidecar inconsistent with weights → user copied the wrong .json.
    //  3. Factory mis-routed the load → e.g. someone passed a shogi
    //     checkpoint as `--weights` to the chess engine.  The chess
    //     prefix would resolve to nonexistent tensor names and we'd fail
    //     with an obscure missing-parameter error deep in build_graph;
    //     here we catch it loudly with the right diagnostic.
    //
    // The check is cheap (one tensor lookup, two shape compares) so it
    // runs unconditionally even in release builds.
    // -------------------------------------------------------------------------
    {
        const std::string in_proj_key = impl_->prefix + "input_proj.weight";
        auto it_w = impl_->params.find(in_proj_key);
        if (it_w == impl_->params.end()) {
            throw std::runtime_error(
                "MlxBackend: weights file " + weights_path +
                " does not contain '" + in_proj_key +
                "' — sidecar reports game='" + config_.game +
                "' but the safetensors looks like a different game "
                "(check the .json sidecar's `game` field).");
        }
        const int wn = static_cast<int>(mlx_array_ndim(it_w->second));
        if (wn != 2) {
            throw std::runtime_error(
                "MlxBackend: " + in_proj_key + " is not a 2-D tensor "
                "(rank=" + std::to_string(wn) + ").");
        }
        const int rows = static_cast<int>(mlx_array_dim(it_w->second, 0));  // d_model
        const int cols = static_cast<int>(mlx_array_dim(it_w->second, 1));  // feat_dim
        if (rows != config_.d_model || cols != config_.feat_dim) {
            std::ostringstream oss;
            oss << "MlxBackend: sidecar/weights shape mismatch for "
                << in_proj_key << " — sidecar declares ["
                << config_.d_model << ", " << config_.feat_dim
                << "] but the loaded tensor is [" << rows << ", " << cols
                << "].  Likely cause: an old per-step checkpoint whose "
                   "sidecar lacks architecture fields and falls back to "
                   "chess defaults.  Re-export via training.export.export_model() "
                   "or update training/src/training/trainers/supervised.py to "
                   "write feat_dim/num_moves/seq_len into the .json sidecar.";
            throw std::runtime_error(oss.str());
        }

        // Best-effort sanity check on the policy head: confirms num_moves.
        const std::string pol_head_key = impl_->prefix + "policy_head.fc2.weight";
        auto it_p = impl_->params.find(pol_head_key);
        if (it_p != impl_->params.end()) {
            const int p_rows = static_cast<int>(mlx_array_dim(it_p->second, 0));
            if (p_rows != config_.num_moves) {
                std::ostringstream oss;
                oss << "MlxBackend: sidecar/weights num_moves mismatch — "
                    << pol_head_key << " has " << p_rows << " output rows but "
                    << "sidecar declares num_moves=" << config_.num_moves
                    << ".  Re-export the checkpoint with up-to-date sidecar.";
                throw std::runtime_error(oss.str());
            }
        }

        // Cross-check sidecar's S*F against the factory-passed input_size_
        // (the engine's encoder output width).  Mismatch means somebody
        // routed a shogi checkpoint through the chess engine or vice versa.
        const std::size_t sidecar_input =
            static_cast<std::size_t>(config_.seq_len) *
            static_cast<std::size_t>(config_.feat_dim);
        if (input_size_ > 0 && sidecar_input != input_size_) {
            std::cerr << "info string MlxBackend: WARNING — sidecar input "
                      << "size (" << sidecar_input << ") != factory-expected ("
                      << input_size_ << ").  Engine will use the sidecar value; "
                      << "if the GUI is set up wrong (e.g. shogi model "
                      << "loaded by chess_engine), inputs from the encoder "
                      << "won't match and inference will fail.\n";
        }
    }

    // -------------------------------------------------------------------------
    // Compile the forward-pass closure via mlx_compile(shapeless=true).
    //
    // mlx_compile traces Impl::closure_cb once, fuses chains of element-wise
    // ops into single Metal shaders (the same fusion that Python mx.compile()
    // performs), and caches the result.  With shapeless=true, the compiled
    // kernel handles any batch size without re-tracing.
    //
    // This reduces ~200 individual Metal kernel launches per forward pass to
    // ~20 fused kernels, matching Python inference latency.
    // -------------------------------------------------------------------------
    {
        std::lock_guard<std::mutex> lk(mlx_mutex_);
        std::cerr << "info string MlxBackend: compiling forward-pass closure...\n";
        mlx_closure raw_cls = mlx_closure_new_func_payload(
            &Impl::closure_cb,
            impl_.get(),
            nullptr  // no destructor — impl_ is owned by MlxBackend
        );
        mlx_closure compiled = mlx_closure_new();
        if (mlx_compile(&compiled, raw_cls, /*shapeless=*/true) == 0) {
            mlx_closure_set(&impl_->compiled_closure, compiled);
            impl_->has_compiled_closure = true;
            std::cerr << "info string MlxBackend: forward-pass closure compiled.\n";
        } else {
            std::cerr << "info string MlxBackend: mlx_compile failed; "
                         "falling back to uncompiled forward pass.\n";
        }
        mlx_closure_free(compiled);
        mlx_closure_free(raw_cls);
    }

    // -------------------------------------------------------------------------
    // GPU warmup: precompile Metal Pipeline State Objects for common batch sizes.
    //
    // Even with mlx_compile, the Metal PSOs for each unique tensor shape are
    // compiled lazily on first use.  By running dummy forward passes now, we
    // trigger that compilation at startup rather than mid-search.
    // -------------------------------------------------------------------------
    std::cerr << "info string MlxBackend: warming up GPU (precompiling Metal PSOs)...\n";
    {
        std::lock_guard<std::mutex> lk(mlx_mutex_);
        const std::vector<int> warmup_batches = {1, 4, 8, 12, 16, 32};
        // Call via `this->` to disambiguate from the local `input_size`
        // constructor parameter (the override returns config_.seq_len * feat_dim).
        const std::size_t in_sz = this->input_size();
        for (int wb : warmup_batches) {
            try {
                std::vector<float> dummy(static_cast<std::size_t>(wb) * in_sz, 0.0f);
                forward_batch_nolock(wb, dummy);
            } catch (const std::exception& e) {
                std::cerr << "info string MlxBackend: warmup B=" << wb
                          << " failed: " << e.what() << "\n";
                break;
            }
        }
    }
    std::cerr << "info string MlxBackend: GPU warmup complete.\n";
}

MlxBackend::~MlxBackend() = default;

std::string MlxBackend::name() const { return name_; }

// -----------------------------------------------------------------------------
// Internal batched forward pass — caller MUST hold mlx_mutex_.
//
// Processes B inputs packed flat into `flat` (size = B * S * F).
// Returns B NNOutput values.  Throws on error (no fallback here; callers
// wrap with try/catch and decide what to do).
// -----------------------------------------------------------------------------
std::vector<NNOutput> MlxBackend::forward_batch_nolock(
        int B, const std::vector<float>& flat) const {

    auto& impl = *impl_;
    const int S = config_.seq_len;
    const int F = config_.feat_dim;

    // Build the input tensor [B, S, F].
    int input_shape[3] = {B, S, F};
    mlx_array x_in = mlx_array_new_data(
        const_cast<float*>(flat.data()), input_shape, 3, MLX_FLOAT32);

    MlxArr policy_logits, value, moves_left;

    if (impl.has_compiled_closure) {
        // -------------------------------------------------------------------------
        // Fast path: dispatch through the compiled closure.
        //
        // mlx_compile has already traced and fused the forward graph into ~20
        // Metal kernel launches.  We just wrap the input in a vector_array and
        // call the compiled closure; MLX dispatches the cached Metal shaders.
        // -------------------------------------------------------------------------
        mlx_vector_array in_vec  = mlx_vector_array_new_value(x_in);
        mlx_array_free(x_in);  // vector_array took a reference; release ours

        mlx_vector_array out_vec = mlx_vector_array_new();
        const int rc = mlx_closure_apply(&out_vec, impl.compiled_closure, in_vec);
        mlx_vector_array_free(in_vec);

        if (rc == 0 && mlx_vector_array_size(out_vec) == 3) {
            mlx_vector_array_get(policy_logits.ptr(), out_vec, 0);
            mlx_vector_array_get(value.ptr(),         out_vec, 1);
            mlx_vector_array_get(moves_left.ptr(),    out_vec, 2);
            mlx_vector_array_free(out_vec);
        } else {
            mlx_vector_array_free(out_vec);
            // Compiled closure failed — fall through to uncompiled path is not
            // feasible here because x_in is already freed.  Return empty to
            // trigger fallback in the caller.
            throw std::runtime_error("forward_batch_nolock: compiled closure failed");
        }
    } else {
        // -------------------------------------------------------------------------
        // Slow path (first warmup call, or if compile failed): build the graph
        // inline via Impl::closure_cb (same code path as the compiled closure,
        // but without caching — used before mlx_compile has run).
        // -------------------------------------------------------------------------
        MlxArr x_holder(x_in);

        mlx_vector_array in_vec  = mlx_vector_array_new_value(x_holder.arr);
        mlx_vector_array out_vec = mlx_vector_array_new();
        const int rc = Impl::closure_cb(&out_vec, in_vec, impl_.get());
        mlx_vector_array_free(in_vec);

        if (rc == 0 && mlx_vector_array_size(out_vec) == 3) {
            mlx_vector_array_get(policy_logits.ptr(), out_vec, 0);
            mlx_vector_array_get(value.ptr(),         out_vec, 1);
            mlx_vector_array_get(moves_left.ptr(),    out_vec, 2);
        }
        mlx_vector_array_free(out_vec);
    }

    // -------------------------------------------------------------------------
    // Materialise — one GPU synchronization for the whole batch.
    // -------------------------------------------------------------------------
    {
        mlx_array eval_targets[] = {policy_logits.arr, value.arr, moves_left.arr};
        for (auto& t : eval_targets) mlx_array_eval(t);
    }

    const float*       p_data    = mlx_array_data_float32(policy_logits.arr);
    const float*       v_data    = mlx_array_data_float32(value.arr);
    const float*       m_data    = mlx_array_data_float32(moves_left.arr);
    const std::size_t  total_pol = mlx_array_size(policy_logits.arr);
    const std::size_t  p_n_model = (B > 0) ? (total_pol / static_cast<std::size_t>(B)) : 0;
    // p_extent clamps the per-leaf usable policy range to what the model
    // actually outputs.  We deliberately do NOT clip against engine_policy_size_
    // any more — that field was the *engine's* expected width (e.g. 4672) which
    // for compact-1858 models would be wider than the model produces.  Callers
    // index via NNOutput::policy_at(p), which returns 0 for out-of-range p.
    const std::size_t  p_extent  = p_n_model;

    // One contiguous batched logit buffer is materialised here and shared
    // across all NNOutput results via shared_ptr.  The old code allocated
    // engine_policy_size_ floats *per leaf* and then memcpy'd a (typically
    // smaller) slice into each, leaving the tail zero-padded — that's
    // ~120 KB of redundant alloc+zero per batch of 16.  The shared layout
    // does one alloc, one sequential memcpy, then trivially indexes by
    // policy_offset = i * p_n_model.
    std::shared_ptr<std::vector<float>> shared_pol;
    if (p_data && B > 0 && p_n_model > 0) {
        shared_pol = std::make_shared<std::vector<float>>(
            static_cast<std::size_t>(B) * p_n_model);
        std::memcpy(shared_pol->data(),
                    p_data,
                    static_cast<std::size_t>(B) * p_n_model * sizeof(float));
    }

    std::vector<NNOutput> results;
    results.reserve(static_cast<std::size_t>(B));
    for (int i = 0; i < B; ++i) {
        NNOutput r;
        if (shared_pol) {
            r.policy_shared = shared_pol;
            r.policy_offset = static_cast<std::size_t>(i) * p_n_model;
            r.policy_extent = p_extent;
        }
        r.value      = v_data ? std::max(-1.0f, std::min(1.0f, v_data[i])) : 0.0f;
        r.moves_left = m_data ? std::max(0.0f, std::min(400.0f, m_data[i])) : 50.0f;
        results.push_back(std::move(r));
    }
    return results;
}

// -----------------------------------------------------------------------------
// Public single-sample evaluate — delegates to the batched path.
// -----------------------------------------------------------------------------
NNOutput MlxBackend::evaluate(const std::vector<float>& input_tensor) {
    // Accept whatever size the *model* expects (sidecar-driven), and not the
    // factory-passed default — the two coincide for current chess/shogi
    // checkpoints, but the sidecar is the source of truth.
    if (input_tensor.size() != input_size())
        throw std::runtime_error("MlxBackend::evaluate: wrong input size");

    std::lock_guard<std::mutex> lk(mlx_mutex_);

    // Pack into a [1, S, F] flat buffer and forward.
    try {
        auto results = forward_batch_nolock(1, input_tensor);
        if (!results.empty()) return std::move(results[0]);
    } catch (const std::exception& e) {
        std::cerr << "info string MlxBackend::evaluate: exception: " << e.what()
                  << "; returning uniform policy\n";
    }

    NNOutput fallback;
    fallback.policy.assign(policy_size(), 0.0f);
    fallback.value      = 0.0f;
    fallback.moves_left = 50.0f;
    return fallback;
}

// -----------------------------------------------------------------------------
// Public batched evaluate — runs all inputs in one GPU forward pass.
// -----------------------------------------------------------------------------
std::vector<NNOutput> MlxBackend::evaluate_batch(
        const std::vector<std::vector<float>>& inputs) {
    if (inputs.empty()) return {};

    const int B = static_cast<int>(inputs.size());

    // Validate sizes.  Use the sidecar-derived effective size, matching what
    // the model actually consumes.
    const std::size_t in_sz = input_size();
    for (const auto& in : inputs) {
        if (in.size() != in_sz)
            throw std::runtime_error("MlxBackend::evaluate_batch: wrong input size");
    }

    // Build flat buffer: [B * S * F].
    std::vector<float> flat;
    flat.reserve(static_cast<std::size_t>(B) * in_sz);
    for (const auto& in : inputs) flat.insert(flat.end(), in.begin(), in.end());

    std::lock_guard<std::mutex> lk(mlx_mutex_);

    // Timing: measure GPU forward pass latency for diagnostic purposes.
    // Logged via stderr (UCI "info string") only for the first 5 batches.
    static std::atomic<int> g_batch_log_count{0};
    const bool log_this = (g_batch_log_count.fetch_add(1, std::memory_order_relaxed) < 5);
    auto t_start = std::chrono::steady_clock::now();

    try {
        auto result = forward_batch_nolock(B, flat);
        if (log_this) {
            auto t_end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cerr << "info string GPU batch B=" << B
                      << " took " << ms << "ms ("
                      << static_cast<double>(B) / ms * 1000.0 << " pos/sec)\n";
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "info string MlxBackend::evaluate_batch: exception: " << e.what()
                  << "; falling back to sequential B=1 calls\n";
        // Fallback: call forward_batch_nolock with B=1 per sample.
        // NOTE: We already hold mlx_mutex_, so we must NOT call evaluate() here
        // (it would deadlock).  Instead call forward_batch_nolock directly.
        std::vector<NNOutput> results;
        results.reserve(static_cast<std::size_t>(B));
        const std::size_t stride   = in_sz;
        const std::size_t fb_psize = policy_size();
        for (int i = 0; i < B; ++i) {
            try {
                std::vector<float> single(flat.begin() + static_cast<std::ptrdiff_t>(i) * stride,
                                          flat.begin() + static_cast<std::ptrdiff_t>(i + 1) * stride);
                auto r = forward_batch_nolock(1, single);
                if (!r.empty()) results.push_back(std::move(r[0]));
                else { NNOutput fb; fb.policy.assign(fb_psize, 0.0f); results.push_back(fb); }
            } catch (...) {
                NNOutput fb;
                fb.policy.assign(fb_psize, 0.0f);
                fb.value = 0.0f; fb.moves_left = 50.0f;
                results.push_back(std::move(fb));
            }
        }
        return results;
    }
}

} // namespace chess_mlx::nn

#else // !CHESS_MLX_HAS_MLX

// Provide stub definitions so the class is still usable at link-time — the
// constructor unconditionally throws.
namespace chess_mlx::nn {

struct MlxBackend::Impl {};

MlxBackend::MlxBackend(const std::string&, std::size_t, std::size_t) {
    throw std::runtime_error("MlxBackend: compiled without CHESS_MLX_HAS_MLX");
}
MlxBackend::~MlxBackend() = default;

NNOutput MlxBackend::evaluate(const std::vector<float>&) {
    throw std::runtime_error("MlxBackend: not available");
}
std::vector<NNOutput> MlxBackend::evaluate_batch(const std::vector<std::vector<float>>&) {
    throw std::runtime_error("MlxBackend: not available");
}
std::string MlxBackend::name() const { return "mlx-unavailable"; }

} // namespace chess_mlx::nn

#endif // CHESS_MLX_HAS_MLX
