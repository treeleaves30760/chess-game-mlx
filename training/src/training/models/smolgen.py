"""
Smolgen module — dynamic attention bias generation.

Inspired by LC0 BT4 (https://lczero.org/blog/2024/02/transformer-progress/).

The BT4 Smolgen generates a *global-context-dependent* attention bias that is
added to the raw QK logits before softmax.  The bias matrix is computed from
a compact bottleneck representation of the entire position.

Architecture:
    1. Per-square linear compression:  d_model → inner_size  (weight shared
       across squares)
    2. Flatten + FC: inner_size*seq_len → gen_size  (global context capture)
       — in practice we use mean-pool then FC to keep params small
    3. FC: gen_size → seq_len * seq_len  (generates one shared bias matrix)
    4. Reshape to [B, seq_len, seq_len] and broadcast to all heads.

Parameter count with defaults (inner=32, gen=64, seq_len=64):
    compress:  d_model × inner = 512×32 = 16,384
    fc1:       inner × gen   = 32×64  =  2,048
    fc2:       gen × (S×S)   = 64×4096 = 262,144  per layer
    Total per layer ~280 K — acceptable for 12-layer model.

For shogi (seq_len=81):
    fc2: 64 × (81×81) = 64×6561 = 420 K per layer — still fine.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Smolgen(nn.Module):
    """Dynamic attention-bias generator (shared across heads).

    Produces a single seq×seq bias matrix per position, broadcast to all
    attention heads.  This keeps parameter count small while still capturing
    global positional context.

    Args:
        d_model:    Model hidden dimension.
        seq_len:    Number of tokens (64 for chess, 81 for shogi).
        n_heads:    Number of attention heads (used for broadcast only).
        inner_size: Per-square compression bottleneck (default 32).
        gen_size:   Global generation bottleneck (default 64).
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        n_heads: int,
        inner_size: int = 16,
        gen_size: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_heads = n_heads

        # Step 1: per-square compression (applied identically to every token)
        self.compress = nn.Linear(d_model, inner_size)

        # Step 2: global pool → FC
        self.fc1 = nn.Linear(inner_size, gen_size)

        # Step 3: generate the flat seq×seq bias matrix
        self.fc2 = nn.Linear(gen_size, seq_len * seq_len)

        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        """Compute dynamic attention bias.

        Args:
            x: Token embeddings `[B, seq_len, d_model]`.

        Returns:
            Bias tensor `[B, n_heads, seq_len, seq_len]` broadcast-filled from
            a single shared `[B, seq_len, seq_len]` matrix.
        """
        B = x.shape[0]

        # [B, seq_len, inner_size]
        compressed = self.act(self.compress(x))

        # [B, inner_size] — global mean pool
        pooled = compressed.mean(axis=1)

        # [B, gen_size]
        gen = self.act(self.fc1(pooled))

        # [B, seq_len * seq_len]
        bias_flat = self.fc2(gen)

        # [B, seq_len, seq_len] → broadcast over heads
        bias = bias_flat.reshape(B, 1, self.seq_len, self.seq_len)

        # [B, n_heads, seq_len, seq_len] via broadcast
        return mx.broadcast_to(bias, (B, self.n_heads, self.seq_len, self.seq_len))
