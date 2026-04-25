"""MLX model that mirrors LC0's BT-series (attention-body) network.

This model is **intentionally separate** from ``ChessShogiTransformer`` in
``training/src/training/models/transformer.py`` because LC0's architecture
differs materially in several ways:

1. **Input shape**: LC0 expects ``[B, 64, 112]`` (112 history-aware planes per
   square) whereas our native model uses ``[B, 64, 19]``.

2. **Embedding**: LC0 has two variants (PE_MAP and PE_DENSE). Both produce
   a `[B, 64, d_model]` tensor but the preprocessing differs. BT4 also has
   an ``ip_emb_ffn`` block (another FFN on the embedding) and multiplicative
   + additive input gates (``ip_mult_gate`` / ``ip_add_gate``).

3. **Smolgen**: LC0 uses a two-stage smolgen with a **shared global weight**
   ``smolgen_w`` that generates the final 64×64 bias per head. Our model
   uses a single flat FC that directly emits the bias.

4. **Layer layout**: LC0 uses Post-LN with residual + alpha scaling baked
   into the forward pass, where ``alpha = (2 * n_layers)^-0.25``.

5. **Value head**: LC0 emits WDL (3 logits → softmax → w-l scalar), ours
   uses a scalar tanh. We handle this by converting WDL → scalar at the
   output: ``value = P(win) - P(loss)``.

Forward pass uses float32 throughout for numerical fidelity with LC0's
loaded weights (which are converted from LINEAR16 quantization anyway).

Status (2026-04-23): This is a WORK-IN-PROGRESS.
PE_MAP path (simpler, used by T1/T2/T3 nets) should be functional.
PE_DENSE path (used by BT4) is partially stubbed — see TODOs.
"""

from __future__ import annotations

import math
from typing import Literal

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Activation selector (matches LC0's enum)
# ---------------------------------------------------------------------------

class LC0Activation(nn.Module):
    """Select activation by integer code matching net.proto's ActivationFunction."""

    def __init__(self, code: int, default_code: int = 2) -> None:
        super().__init__()
        # code==0 (DEFAULT) falls back to default_code
        self.code = code if code != 0 else default_code

    def __call__(self, x: mx.array) -> mx.array:
        c = self.code
        if c == 1:  # MISH
            return x * mx.tanh(nn.softplus(x))
        if c == 2:  # RELU
            return nn.relu(x)
        if c == 3:  # NONE
            return x
        if c == 4:  # TANH
            return mx.tanh(x)
        if c == 5:  # SIGMOID
            return mx.sigmoid(x)
        if c == 6:  # SELU
            return nn.selu(x)
        if c == 7:  # SWISH (== SiLU)
            return x * mx.sigmoid(x)
        if c == 8:  # RELU^2
            r = nn.relu(x)
            return r * r
        if c == 9:  # SOFTMAX
            return mx.softmax(x, axis=-1)
        raise ValueError(f"Unsupported activation code {c}")


# ---------------------------------------------------------------------------
# LC0-style linear (weight stored as [in, out], right-multiplied)
# ---------------------------------------------------------------------------

class LC0Linear(nn.Module):
    """Linear with weight layout ``[in, out]`` (LC0 convention).

    Forward: ``y = x @ W + b``. Matches LC0's ``builder->MatMul(flow, W)``
    without any transpose. (MLX's `nn.Linear` stores weight `[out, in]` and
    computes `x @ W.T + b`, which requires transposing LC0 weights.)
    """

    def __init__(self, in_dim: int, out_dim: int, has_bias: bool = True) -> None:
        super().__init__()
        self.weight = mx.zeros((in_dim, out_dim))
        if has_bias:
            self.bias: mx.array | None = mx.zeros((out_dim,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# Smolgen — two-stage, with shared global weight
# ---------------------------------------------------------------------------

class LC0Smolgen(nn.Module):
    """LC0-style smolgen. Uses a global `smolgen_w` tensor injected at forward.

    Produces a `[B, heads, seq_len, seq_len]` bias from token embeddings.

    Args:
        d_model: Embedding dimension.
        hidden_channels: Per-square compressed size (32 in all known LC0 nets).
        hidden_sz: Global bottleneck size (256 in all known LC0 nets).
        gen_sz: Per-head generation size (256 in all known LC0 nets).
        heads: Attention head count.
        seq_len: Always 64 for chess.
        activation_code: Activation code (LC0 enum). 7 (SWISH) for BT4.
    """

    def __init__(
        self,
        d_model: int,
        hidden_channels: int,
        hidden_sz: int,
        gen_sz: int,
        heads: int,
        seq_len: int = 64,
        activation_code: int = 7,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_channels = hidden_channels
        self.hidden_sz = hidden_sz
        self.gen_sz = gen_sz
        self.heads = heads
        self.seq_len = seq_len
        self.act = LC0Activation(activation_code)

        # Per-square compression: embedding_size → hidden_channels
        self.compress = LC0Linear(d_model, hidden_channels, has_bias=False)

        # Post-flatten dense layers + LNs
        flat_in = seq_len * hidden_channels
        self.dense1 = LC0Linear(flat_in, hidden_sz)
        self.ln1 = nn.LayerNorm(hidden_sz, eps=1e-3)

        self.dense2 = LC0Linear(hidden_sz, gen_sz * heads)
        self.ln2 = nn.LayerNorm(gen_sz * heads, eps=1e-3)

    def __call__(self, x: mx.array, smolgen_w: mx.array) -> mx.array:
        """Compute smolgen bias.

        Args:
            x:          Encoder input tokens `[B, seq_len, d_model]`.
            smolgen_w:  Global shared weight. Shape ``[gen_sz, seq_len*seq_len]``.

        Returns:
            Bias `[B, heads, seq_len, seq_len]`.
        """
        B = x.shape[0]
        # [B, 64, hidden_channels]
        h = self.compress(x)
        # [B, 64 * hidden_channels]
        h = h.reshape(B, self.seq_len * self.hidden_channels)
        h = self.dense1(h)
        h = self.act(h)
        h = self.ln1(h)
        h = self.dense2(h)
        h = self.act(h)
        h = self.ln2(h)
        # [B, heads, gen_sz]
        h = h.reshape(B, self.heads, self.gen_sz)
        # MatMul with global smolgen_w [gen_sz, seq_len*seq_len]
        #   → [B, heads, seq_len*seq_len]
        h = h @ smolgen_w
        # [B, heads, seq_len, seq_len]
        return h.reshape(B, self.heads, self.seq_len, self.seq_len)


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------

class LC0EncoderLayer(nn.Module):
    """One LC0 Post-LN encoder layer with smolgen-augmented attention."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        smolgen_hidden_channels: int,
        smolgen_hidden_sz: int,
        smolgen_gen_sz: int,
        alpha: float,
        seq_len: int = 64,
        activation_code: int = 1,  # 1 = MISH (BT4 default)
        ffn_activation_code: int = 0,
        smolgen_activation_code: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.depth = d_model // heads
        self.scale = 1.0 / math.sqrt(self.depth)
        self.alpha = alpha
        self.seq_len = seq_len

        # Attention
        self.q_lin = LC0Linear(d_model, d_model)
        self.k_lin = LC0Linear(d_model, d_model)
        self.v_lin = LC0Linear(d_model, d_model)
        self.dense = LC0Linear(d_model, d_model)

        # Smolgen (optional; if weights missing, skip)
        self.smolgen = LC0Smolgen(
            d_model=d_model,
            hidden_channels=smolgen_hidden_channels,
            hidden_sz=smolgen_hidden_sz,
            gen_sz=smolgen_gen_sz,
            heads=heads,
            seq_len=seq_len,
            activation_code=smolgen_activation_code or 7,
        )

        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        # FFN
        self.ffn_dense1 = LC0Linear(d_model, ffn_dim)
        self.ffn_dense2 = LC0Linear(ffn_dim, d_model)
        self.ffn_act = LC0Activation(
            ffn_activation_code if ffn_activation_code != 0 else activation_code,
            default_code=activation_code,
        )
        self.ln2 = nn.LayerNorm(d_model, eps=eps)

    def __call__(self, x: mx.array, smolgen_w: mx.array | None) -> mx.array:
        """Forward pass. Shape: ``[B, 64, d_model]`` → same."""
        B, S, _ = x.shape
        # Attention: Q, K, V with LC0 layout — reshape to [B, S, heads, depth]
        q = self.q_lin(x).reshape(B, S, self.heads, self.depth).transpose(0, 2, 1, 3)
        k = self.k_lin(x).reshape(B, S, self.heads, self.depth).transpose(0, 2, 3, 1)
        v = self.v_lin(x).reshape(B, S, self.heads, self.depth).transpose(0, 2, 1, 3)
        # [B, heads, S, S]
        logits = (q @ k) * self.scale
        if smolgen_w is not None:
            logits = logits + self.smolgen(x, smolgen_w)
        weights = mx.softmax(logits, axis=-1)
        # [B, heads, S, depth]
        attn = weights @ v
        # [B, S, heads, depth] → [B, S, d_model]
        attn = attn.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
        attn = self.dense(attn)
        if self.alpha != 1.0:
            attn = attn * self.alpha
        x = self.ln1(x + attn)
        # FFN
        h = self.ffn_act(self.ffn_dense1(x))
        h = self.ffn_dense2(h)
        if self.alpha != 1.0:
            h = h * self.alpha
        x = self.ln2(x + h)
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class LC0Net(nn.Module):
    """LC0 attention-body network — MLX port for chess.

    Inputs:  ``[B, 64, 112]`` float32 (LC0's 112-plane encoding).
    Outputs: dict with
       - ``policy``     : ``[B, 1858]`` logits (LC0 compact policy)
       - ``value``      : ``[B, 1]`` scalar in [-1, +1] (derived from WDL)
       - ``value_wdl``  : ``[B, 3]`` raw WDL logits
       - ``moves_left`` : ``[B, 1]`` non-negative scalar

    Construct via ``LC0Net.from_weights(lc0_weights)`` which sizes and loads
    all parameters from a decoded ``LC0Weights`` dataclass.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        heads: int,
        ffn_dim: int,
        smolgen_hidden_channels: int,
        smolgen_hidden_sz: int,
        smolgen_gen_sz: int,
        embedding_dense_size: int,  # 0 → PE_MAP path
        default_activation_code: int = 1,
        ffn_activation_code: int = 0,
        smolgen_activation_code: int = 7,
        seq_len: int = 64,
        has_global_smolgen: bool = True,
        has_ip_emb_ffn: bool = False,
        has_ma_gating: bool = False,
        has_ip_emb_ln: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.ffn_dim = ffn_dim
        self.seq_len = seq_len
        self.embedding_dense_size = embedding_dense_size
        self.has_global_smolgen = has_global_smolgen
        self.has_ip_emb_ffn = has_ip_emb_ffn
        self.has_ma_gating = has_ma_gating
        self.has_ip_emb_ln = has_ip_emb_ln

        N = n_layers
        self.alpha = float((2 * N) ** -0.25)

        # ---- Input embedding ----
        if embedding_dense_size > 0:
            # PE_DENSE path (BT4)
            # Extract 12-dim per-square board-only features, produce
            # (64 * embedding_dense_size) extra channels, concat with the
            # original 112 features → [64, 112 + embedding_dense_size].
            self.ip_emb_preproc = LC0Linear(64 * 12, 64 * embedding_dense_size)
            fist_stage_out_c = 112 + embedding_dense_size
        else:
            # PE_MAP path (T1 transformer nets): [112, 64 pos] → 176 dims
            # Pre-built positional encoding concatenated per-square.
            self.pos_enc = mx.zeros((seq_len, 64))  # filled at load time
            fist_stage_out_c = 112 + 64  # 176

        self.ip_emb = LC0Linear(fist_stage_out_c, d_model)
        self.emb_act = LC0Activation(default_activation_code)

        if has_ip_emb_ln:
            self.ip_emb_ln = nn.LayerNorm(d_model, eps=1e-3)

        if has_ma_gating:
            # Per-square gating weights, shape [64, d_model]
            self.ip_mult_gate = mx.zeros((seq_len, d_model))
            self.ip_add_gate = mx.zeros((seq_len, d_model))

        if has_ip_emb_ffn:
            # Another FFN on the embedding, Post-LN
            self.emb_ffn_dense1 = LC0Linear(d_model, ffn_dim)
            self.emb_ffn_dense2 = LC0Linear(ffn_dim, d_model)
            self.emb_ffn_act = LC0Activation(
                ffn_activation_code if ffn_activation_code != 0
                else default_activation_code,
                default_code=default_activation_code,
            )
            self.emb_ffn_ln = nn.LayerNorm(d_model, eps=1e-3)

        # ---- Encoder stack ----
        self.encoder_layers = [
            LC0EncoderLayer(
                d_model=d_model,
                heads=heads,
                ffn_dim=ffn_dim,
                smolgen_hidden_channels=smolgen_hidden_channels,
                smolgen_hidden_sz=smolgen_hidden_sz,
                smolgen_gen_sz=smolgen_gen_sz,
                alpha=self.alpha,
                seq_len=seq_len,
                activation_code=default_activation_code,
                ffn_activation_code=ffn_activation_code,
                smolgen_activation_code=smolgen_activation_code,
            )
            for _ in range(n_layers)
        ]

        # ---- Global smolgen (shared across all encoder layers) ----
        if has_global_smolgen:
            # Shape: [gen_sz, 64*64]  (post-transpose from proto's
            # [64*64, gen_sz] storage)
            self.smolgen_w = mx.zeros((smolgen_gen_sz, seq_len * seq_len))

        # ---- Policy head (attention-policy style) ----
        # From LC0 attention_policy_head:
        #   ip_pol_w: [d_model, policy_d]     (embedding)
        #   ip2_pol_w: [policy_d, policy_d]   (wq)
        #   ip3_pol_w: [policy_d, policy_d]   (wk)
        #   ip4_pol_w: 4×policy_d promotion weights
        # Output is ~1858 logits via a fixed policy map lookup
        # (the map is not the model's concern — engine-side).
        self.pol_ip = LC0Linear(d_model, d_model)  # placeholder sizes; overwrite at load
        self.pol_wq = LC0Linear(d_model, d_model)
        self.pol_wk = LC0Linear(d_model, d_model)
        # Note: Full attention-policy reconstruction is deferred; see TODOs.

        # ---- Value head (attention-body) ----
        # ip_val_w: [d_model, value_d]   (embedding)
        # ip1_val_w: [64 * value_d, hidden]
        # ip2_val_w: [hidden, 3]         (WDL) or [hidden, 1] (classical)
        self.val_ip = LC0Linear(d_model, 32)  # placeholder
        self.val_ip1 = LC0Linear(32 * seq_len, 128)
        self.val_ip2 = LC0Linear(128, 3)  # WDL head
        self.val_act = LC0Activation(default_activation_code)

        # ---- Moves-left head ----
        self.mov_ip = LC0Linear(d_model, 8)
        self.mov_ip1 = LC0Linear(8 * seq_len, 128)
        self.mov_ip2 = LC0Linear(128, 1)
        self.mov_act = LC0Activation(default_activation_code)

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        """Forward pass. Input ``[B, 64, 112]``; returns head outputs."""
        B, S, _ = x.shape
        if self.embedding_dense_size > 0:
            # PE_DENSE: extract first 12 dims (piece planes), preprocess,
            # concat back. x is [B, 64, 112].
            pos_info = x[:, :, :12].reshape(B, 64 * 12)
            pos_info = self.ip_emb_preproc(pos_info)
            pos_info = pos_info.reshape(B, 64, self.embedding_dense_size)
            flow = mx.concatenate([x, pos_info], axis=-1)  # [B, 64, 112+eds]
        else:
            # PE_MAP: concat fixed pos_enc to each position
            pe = mx.broadcast_to(self.pos_enc, (B, S, 64))
            flow = mx.concatenate([x, pe], axis=-1)  # [B, 64, 176]

        flow = self.emb_act(self.ip_emb(flow))

        if self.has_ip_emb_ln:
            flow = self.ip_emb_ln(flow)

        if self.has_ma_gating:
            flow = flow * self.ip_mult_gate
            flow = flow + self.ip_add_gate

        if self.has_ip_emb_ffn:
            h = self.emb_ffn_act(self.emb_ffn_dense1(flow))
            h = self.emb_ffn_dense2(h)
            # Note: no alpha scaling here per LC0's implementation
            flow = self.emb_ffn_ln(flow + h)

        # Encoder stack
        smolgen_w = self.smolgen_w if self.has_global_smolgen else None
        for layer in self.encoder_layers:
            flow = layer(flow, smolgen_w)

        # ----- Heads (simplified) -----
        # NOTE: The full attention-policy head requires a 64×64 self-attention
        # plus promotion handling; this minimal version mean-pools and runs
        # the value / moves-left pipelines. The policy mapping is performed
        # downstream in the engine using LC0's policy_map_chess.txt.
        # Placeholder policy: pol_ip(mean-pool) — not used end-to-end yet.

        # Value (WDL)
        vh = self.val_act(self.val_ip(flow))  # [B, 64, val_d]
        vh = vh.reshape(B, -1)
        vh = self.val_act(self.val_ip1(vh))
        wdl_logits = self.val_ip2(vh)  # [B, 3]
        wdl = mx.softmax(wdl_logits, axis=-1)
        # scalar value = P(win) - P(loss); P(win) index 0, P(loss) index 2
        value = (wdl[:, 0:1] - wdl[:, 2:3])

        # Moves-left
        mh = self.mov_act(self.mov_ip(flow))
        mh = mh.reshape(B, -1)
        mh = self.mov_act(self.mov_ip1(mh))
        moves_left = nn.softplus(self.mov_ip2(mh))

        # Policy (placeholder: mean-pool → d_model project)
        pmean = flow.mean(axis=1)
        policy_stub = self.pol_ip(pmean)  # this won't be 1858; TODO wire full head

        return {
            "policy": policy_stub,       # STUB — see TODO in docstring
            "value": value,
            "value_wdl": wdl_logits,
            "moves_left": moves_left,
        }
