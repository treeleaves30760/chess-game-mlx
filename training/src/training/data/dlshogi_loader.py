"""
dlshogi-compatible data loader for shogi positions.

Currently provides a **synthetic** loader that generates random shogi positions
using python-shogi and assigns random action-values.  The real dlshogi teacher
data loader can be slotted in later with the same batch interface.

Batch format (same convention as chessbench_loader):
    positions:      [B, 81, 90]  bf16
    policy_target:  [B, 2187]    float32 soft labels / one-hot
    value_target:   [B, 1]       float32 in [-1, +1]
    moves_left:     [B, 1]       float32 ≥ 0
"""

from __future__ import annotations

import random
from collections.abc import Generator, Iterator
from typing import NamedTuple

import mlx.core as mx
import numpy as np

SHOGI_NUM_MOVES = 2187


class ShogiBatch(NamedTuple):
    """A single mini-batch of shogi training data as MLX arrays."""

    positions: mx.array      # [B, 81, 90]  bf16
    policy_target: mx.array  # [B, 2187]    float32
    value_target: mx.array   # [B, 1]       float32
    moves_left: mx.array     # [B, 1]       float32


def _try_encode_shogi():  # type: ignore[return]
    """Import shogi encoder; fall back to None if python-shogi unavailable."""
    try:
        from training.data.encoding import encode_shogi_position  # noqa: PLC0415
        return encode_shogi_position
    except ImportError:
        return None


def synthetic_shogi_batch_generator(
    batch_size: int = 32,
    seed: int | None = None,
    max_batches: int | None = None,
) -> Generator[ShogiBatch, None, None]:
    """Infinite (or bounded) generator of synthetic shogi training batches.

    If python-shogi is not installed, generates random tensors with correct
    shapes so the training pipeline can still be tested end-to-end.

    Args:
        batch_size:  Positions per batch.
        seed:        Optional RNG seed.
        max_batches: Stop after this many batches (``None`` = infinite).

    Yields:
        ``ShogiBatch`` namedtuples.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    encode_fn = _try_encode_shogi()

    batches_produced = 0

    while max_batches is None or batches_produced < max_batches:
        if encode_fn is not None:
            try:
                import shogi  # type: ignore[import]  # noqa: PLC0415

                positions_list: list[np.ndarray] = []
                policy_indices: list[int] = []
                values: list[float] = []
                moves_left_vals: list[float] = []

                while len(positions_list) < batch_size:
                    board = shogi.Board()
                    for _ in range(rng.randint(0, 60)):
                        moves = list(board.legal_moves)
                        if not moves or board.is_game_over():
                            break
                        mv = rng.choice(moves)
                        board.push(mv)

                    positions_list.append(encode_fn(board))
                    policy_indices.append(rng.randint(0, SHOGI_NUM_MOVES - 1))
                    values.append(float(np_rng.uniform(-1.0, 1.0)))
                    moves_left_vals.append(float(np_rng.uniform(1, 200)))

                positions_np = np.stack(positions_list[:batch_size])
                policy_np = np.zeros((batch_size, SHOGI_NUM_MOVES), dtype=np.float32)
                for i, idx in enumerate(policy_indices[:batch_size]):
                    policy_np[i, idx] = 1.0
                value_np = np.array([[v] for v in values[:batch_size]], dtype=np.float32)
                ml_np = np.array([[m] for m in moves_left_vals[:batch_size]], dtype=np.float32)

            except (ImportError, Exception):
                # Fallback: pure random tensors
                positions_np = np_rng.random((batch_size, 81, 90)).astype(np.float32)
                policy_np = np.zeros((batch_size, SHOGI_NUM_MOVES), dtype=np.float32)
                rand_idx = np_rng.integers(0, SHOGI_NUM_MOVES, size=batch_size)
                for i, idx in enumerate(rand_idx):
                    policy_np[i, idx] = 1.0
                value_np = np_rng.uniform(-1.0, 1.0, (batch_size, 1)).astype(np.float32)
                ml_np = np_rng.uniform(1, 200, (batch_size, 1)).astype(np.float32)
        else:
            # python-shogi not installed — random tensors
            positions_np = np_rng.random((batch_size, 81, 90)).astype(np.float32)
            policy_np = np.zeros((batch_size, SHOGI_NUM_MOVES), dtype=np.float32)
            rand_idx = np_rng.integers(0, SHOGI_NUM_MOVES, size=batch_size)
            for i, idx in enumerate(rand_idx):
                policy_np[i, idx] = 1.0
            value_np = np_rng.uniform(-1.0, 1.0, (batch_size, 1)).astype(np.float32)
            ml_np = np_rng.uniform(1, 200, (batch_size, 1)).astype(np.float32)

        yield ShogiBatch(
            positions=mx.array(positions_np, dtype=mx.bfloat16),
            policy_target=mx.array(policy_np),
            value_target=mx.array(value_np),
            moves_left=mx.array(ml_np),
        )
        batches_produced += 1


class DlshogiLoader:
    """Stub loader for real dlshogi teacher data.

    Args:
        data_dir:   Directory containing dlshogi ``.hcpe`` files.
        batch_size: Mini-batch size.
    """

    def __init__(self, data_dir: str, batch_size: int = 256) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[ShogiBatch]:
        raise NotImplementedError(
            "Real dlshogi loader not yet implemented.  "
            "Use synthetic_shogi_batch_generator() for now."
        )


def make_shogi_loader(
    synthetic: bool = True,
    data_dir: str | None = None,
    batch_size: int = 256,
    seed: int = 42,
    max_batches: int | None = None,
) -> Generator[ShogiBatch, None, None] | Iterator[ShogiBatch]:
    """Create a shogi data loader."""
    if synthetic:
        return synthetic_shogi_batch_generator(
            batch_size=batch_size, seed=seed, max_batches=max_batches
        )
    if data_dir is None:
        raise ValueError("data_dir must be set when synthetic=False")
    return DlshogiLoader(data_dir=data_dir, batch_size=batch_size)
