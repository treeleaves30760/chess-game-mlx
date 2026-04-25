"""
Output heads for ChessShogiTransformer.

Three heads:
- PolicyHead:     [B, seq_len, d_model] → [B, num_moves]   (logits, no softmax)
- ValueHead:      [B, seq_len, d_model] → [B, 1]            (tanh output, [-1,+1])
- MovesLeftHead:  [B, seq_len, d_model] → [B, 1]            (softplus, ≥0)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class PolicyHead(nn.Module):
    """Policy head — outputs unnormalised logits over the move vocabulary.

    Uses a two-layer MLP on the mean-pooled token representation.

    Args:
        d_model: Input hidden dimension.
        num_moves: Size of the move vocabulary (1858 for chess, 2187 for shogi).
    """

    def __init__(self, d_model: int, num_moves: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_moves)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: `[B, seq_len, d_model]`

        Returns:
            `[B, num_moves]` logits.
        """
        # Mean pool over sequence dimension
        h = x.mean(axis=1)  # [B, d_model]
        h = self.norm(h)
        h = self.act(self.fc1(h))
        return self.fc2(h)  # [B, num_moves]


class ValueHead(nn.Module):
    """Value head — predicts game outcome in [-1, +1] via tanh.

    Positive = white/先手 advantage (convention from encoding_spec §2.3).

    Args:
        d_model: Input hidden dimension.
        hidden_size: Intermediate projection size (default 256).
    """

    def __init__(self, d_model: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: `[B, seq_len, d_model]`

        Returns:
            `[B, 1]` values in `[-1, +1]`.
        """
        h = x.mean(axis=1)  # [B, d_model]
        h = self.norm(h)
        h = self.act(self.fc1(h))
        return mx.tanh(self.fc2(h))  # [B, 1]


class MovesLeftHead(nn.Module):
    """Moves-left auxiliary head — predicts remaining half-moves via softplus.

    Helps stabilise training (per AlphaZero practice).

    Args:
        d_model: Input hidden dimension.
        hidden_size: Intermediate projection size (default 128).
    """

    def __init__(self, d_model: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: `[B, seq_len, d_model]`

        Returns:
            `[B, 1]` non-negative scalar (half-moves remaining).
        """
        h = x.mean(axis=1)  # [B, d_model]
        h = self.norm(h)
        h = self.act(self.fc1(h))
        raw = self.fc2(h)  # [B, 1]
        return nn.softplus(raw)  # [B, 1]
