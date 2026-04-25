"""
ChessBench-compatible data loader for chess positions.

Two modes:
1. **Synthetic** (default, no download required): generates random chess
   positions by self-play with python-chess and assigns random action-values.
   Suitable for end-to-end pipeline testing.

2. **Real ChessBench** (drop-in, same interface): streaming loader for the
   DeepMind ChessBench bagz files once downloaded.  Activate by passing
   ``synthetic=False`` and a valid ``data_dir``.

Batch format (MLX arrays, bfloat16 on M3):
    positions:      [B, 64, 19]  float32 → cast to bf16 at loader boundary
    policy_target:  [B, 1858]    float32 soft labels (or one-hot)
    value_target:   [B, 1]       float32 in [-1, +1]
    moves_left:     [B, 1]       float32 ≥ 0
"""

from __future__ import annotations

import random
from collections.abc import Generator, Iterator
from typing import NamedTuple

import chess
import chess.pgn
import mlx.core as mx
import numpy as np

from training.data.encoding import (
    _CHESS_NUM_COMPACT,
    chess_move_to_idx,
    encode_chess_position,
)

# Re-export so callers can use this module's constant
CHESS_NUM_MOVES = _CHESS_NUM_COMPACT


class ChessBatch(NamedTuple):
    """A single mini-batch of chess training data as MLX arrays."""

    positions: mx.array      # [B, 64, 19]  bf16
    policy_target: mx.array  # [B, 1858]    float32
    value_target: mx.array   # [B, 1]       float32
    moves_left: mx.array     # [B, 1]       float32


# ---------------------------------------------------------------------------
# Synthetic position generator
# ---------------------------------------------------------------------------

def _random_game_positions(
    max_positions: int = 200,
    rng: random.Random | None = None,
) -> list[tuple[chess.Board, int | None, float, float]]:
    """Generate positions from a random self-play game.

    Returns a list of (board, move_idx_or_None, value, moves_left) tuples.
    """
    if rng is None:
        rng = random.Random()

    board = chess.Board()
    records: list[tuple[chess.Board, int | None, float, float]] = []
    total_half_moves = rng.randint(20, 200)

    for half_move in range(total_half_moves):
        if board.is_game_over():
            break

        legal = list(board.legal_moves)
        if not legal:
            break

        move = rng.choice(legal)

        # Encode position before making the move
        try:
            policy_idx = chess_move_to_idx(move, board)
        except (KeyError, ValueError):
            policy_idx = None

        # Fake value: random in [-1, +1] slightly biased by material count
        material = sum(
            len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        )
        # Simple material heuristic normalised by max ~39
        fake_value = float(np.clip(material / 39.0 + rng.gauss(0, 0.3), -1.0, 1.0))
        moves_left = float(max(1, total_half_moves - half_move))

        records.append((board.copy(), policy_idx, fake_value, moves_left))
        board.push(move)

        if len(records) >= max_positions:
            break

    return records


def synthetic_chess_batch_generator(
    batch_size: int = 32,
    seed: int | None = None,
    max_batches: int | None = None,
) -> Generator[ChessBatch, None, None]:
    """Infinite (or bounded) generator of synthetic chess training batches.

    Args:
        batch_size:  Number of positions per batch.
        seed:        Optional RNG seed for reproducibility.
        max_batches: If set, stop after this many batches.

    Yields:
        ``ChessBatch`` namedtuples with MLX arrays.
    """
    rng = random.Random(seed)
    batches_produced = 0
    buffer: list[tuple[chess.Board, int | None, float, float]] = []

    while max_batches is None or batches_produced < max_batches:
        # Refill buffer
        while len(buffer) < batch_size:
            buffer.extend(_random_game_positions(rng=rng))

        batch_records = buffer[:batch_size]
        buffer = buffer[batch_size:]

        # Build numpy arrays
        positions_np = np.stack([encode_chess_position(rec[0]) for rec in batch_records])  # [B, 64, 19]
        policy_np = np.zeros((batch_size, _CHESS_NUM_COMPACT), dtype=np.float32)
        for i, (_, idx, _, _) in enumerate(batch_records):
            if idx is not None and 0 <= idx < _CHESS_NUM_COMPACT:
                policy_np[i, idx] = 1.0
            # If no valid index: leave as uniform zero (model will learn nothing from it)

        value_np = np.array([[rec[2]] for rec in batch_records], dtype=np.float32)
        moves_left_np = np.array([[rec[3]] for rec in batch_records], dtype=np.float32)

        yield ChessBatch(
            positions=mx.array(positions_np, dtype=mx.bfloat16),
            policy_target=mx.array(policy_np),
            value_target=mx.array(value_np),
            moves_left=mx.array(moves_left_np),
        )
        batches_produced += 1


# ---------------------------------------------------------------------------
# Real ChessBench loader (stub — same interface)
# ---------------------------------------------------------------------------

class ChessBenchLoader:
    """Streaming loader for real ChessBench bagz files.

    Drop-in replacement for synthetic generator once data is downloaded.

    Args:
        data_dir:   Path to directory containing ``.bagz`` files.
        batch_size: Mini-batch size.
        shuffle:    Whether to shuffle within shards.
        seed:       RNG seed.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[ChessBatch]:
        raise NotImplementedError(
            "Real ChessBench loader not yet implemented.  "
            "Use synthetic_chess_batch_generator() for now."
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_chess_loader(
    synthetic: bool = True,
    data_dir: str | None = None,
    batch_size: int = 256,
    seed: int = 42,
    max_batches: int | None = None,
) -> Generator[ChessBatch, None, None] | Iterator[ChessBatch]:
    """Create a chess data loader.

    Args:
        synthetic:   Use synthetic random data (default ``True``).
        data_dir:    Required when ``synthetic=False``.
        batch_size:  Mini-batch size.
        seed:        RNG seed.
        max_batches: Limit on number of batches (synthetic only).

    Returns:
        An iterable that yields ``ChessBatch`` objects.
    """
    if synthetic:
        return synthetic_chess_batch_generator(
            batch_size=batch_size, seed=seed, max_batches=max_batches
        )
    if data_dir is None:
        raise ValueError("data_dir must be set when synthetic=False")
    return ChessBenchLoader(data_dir=data_dir, batch_size=batch_size, seed=seed)
