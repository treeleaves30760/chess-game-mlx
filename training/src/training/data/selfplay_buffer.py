"""
In-memory ring buffer for self-play positions.

Used during the self-play fine-tuning phase (Phase 7+).
Currently a minimal implementation; wired up in Phase 7.

The buffer stores positions as numpy arrays and samples random mini-batches.
Once full, old entries are overwritten in FIFO order (ring buffer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import mlx.core as mx
import numpy as np


@dataclass
class SelfPlayEntry:
    """Single position stored in the self-play replay buffer."""

    position: np.ndarray    # [seq_len, feat_dim] float32
    policy: np.ndarray      # [num_moves] float32 soft labels
    value: float            # scalar in [-1, +1]
    moves_left: float       # half-moves remaining
    game: str               # "chess" or "shogi"


class SelfPlayBatch(NamedTuple):
    """A sampled mini-batch from the self-play buffer."""

    positions: mx.array      # [B, seq_len, feat_dim]  bf16
    policy_target: mx.array  # [B, num_moves]           float32
    value_target: mx.array   # [B, 1]                   float32
    moves_left: mx.array     # [B, 1]                   float32


class SelfPlayBuffer:
    """Fixed-capacity ring buffer for self-play positions.

    Args:
        capacity: Maximum number of positions to store.
        seed:     RNG seed for sampling.
    """

    def __init__(self, capacity: int = 1_000_000, seed: int = 42) -> None:
        self.capacity = capacity
        self._entries: list[SelfPlayEntry] = []
        self._write_idx: int = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: SelfPlayEntry) -> None:
        """Add a single position to the buffer."""
        if len(self._entries) < self.capacity:
            self._entries.append(entry)
        else:
            self._entries[self._write_idx] = entry
        self._write_idx = (self._write_idx + 1) % self.capacity

    def add_batch(self, entries: list[SelfPlayEntry]) -> None:
        """Add multiple positions at once."""
        for entry in entries:
            self.add(entry)

    def sample(self, batch_size: int) -> SelfPlayBatch:
        """Sample a random mini-batch.

        Args:
            batch_size: Number of positions to sample.

        Returns:
            ``SelfPlayBatch`` with MLX arrays.

        Raises:
            ValueError: If buffer has fewer entries than ``batch_size``.
        """
        if len(self._entries) < batch_size:
            raise ValueError(
                f"Buffer has only {len(self._entries)} entries, "
                f"but requested batch_size={batch_size}"
            )

        indices = self._rng.choice(len(self._entries), size=batch_size, replace=False)
        samples = [self._entries[i] for i in indices]

        positions_np = np.stack([s.position for s in samples])
        policy_np = np.stack([s.policy for s in samples])
        value_np = np.array([[s.value] for s in samples], dtype=np.float32)
        ml_np = np.array([[s.moves_left] for s in samples], dtype=np.float32)

        return SelfPlayBatch(
            positions=mx.array(positions_np, dtype=mx.bfloat16),
            policy_target=mx.array(policy_np),
            value_target=mx.array(value_np),
            moves_left=mx.array(ml_np),
        )

    @property
    def is_ready(self) -> bool:
        """True if buffer has at least one entry."""
        return len(self._entries) > 0
