"""
Cross-game joint data loader for XGD (Cross-Game Joint Distillation).

Yields alternating chess and shogi batches wrapped in a unified ``JointBatch``
type so the joint trainer can dispatch to the correct model path.

Interleave strategies:
  ``"round_robin"``   — strict chess / shogi / chess / shogi alternation.
  ``"weighted"``      — each step draws a game ~ Bernoulli(p_shogi).

Both game iterators are independent: when one epoch ends the other keeps
going, so unequal dataset sizes never cause a stall.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Literal, NamedTuple

import mlx.core as mx

from training.data.chessbench_loader import ChessBatch
from training.data.dlshogi_loader import ShogiBatch

__all__ = [
    "JointBatch",
    "JointLoader",
]


class JointBatch(NamedTuple):
    """A mixed-game training batch — exactly one game's data per step.

    ``moves_left_target`` may be ``None`` when the teacher data does not
    carry remaining-ply information (e.g. dlshogi hcpe files).  The joint
    trainer passes a zero tensor in that case so the moves-left head still
    receives a gradient-free target, matching single-game behaviour.
    """

    game: Literal["chess", "shogi"]
    positions: mx.array           # [B, 64, 19] chess | [B, 81, 90] shogi  (bf16)
    policy_target: mx.array       # [B, 1858]   chess | [B, 2187]   shogi  (float32)
    value_target: mx.array        # [B, 1]  float32
    moves_left_target: mx.array   # [B, 1]  float32 (zeros when unavailable)


def _chess_batch_to_joint(batch: ChessBatch) -> JointBatch:
    """Wrap a ``ChessBatch`` into a ``JointBatch``."""
    return JointBatch(
        game="chess",
        positions=batch.positions,
        policy_target=batch.policy_target,
        value_target=batch.value_target,
        moves_left_target=batch.moves_left,
    )


def _shogi_batch_to_joint(batch: ShogiBatch) -> JointBatch:
    """Wrap a ``ShogiBatch`` into a ``JointBatch``.

    ``ShogiBatch.moves_left`` is always present (set to 0 by the dlshogi
    loader when hcpe lacks per-position ply data), so no substitution needed.
    """
    return JointBatch(
        game="shogi",
        positions=batch.positions,
        policy_target=batch.policy_target,
        value_target=batch.value_target,
        moves_left_target=batch.moves_left,
    )


class JointLoader:
    """Yields interleaved chess and shogi batches for joint distillation.

    Each game keeps its own independent iterator so unequal epoch lengths
    never block progress: when one iterator is exhausted it wraps around
    automatically (callers typically pass infinite generators anyway).

    Args:
        chess_loader:   An iterator that yields ``ChessBatch`` objects.
        shogi_loader:   An iterator that yields ``ShogiBatch`` objects.
        interleave:     Dispatch strategy.
            ``"round_robin"`` — strict alternation (default).
            ``"weighted"``    — Bernoulli sample; set ``weights`` accordingly.
        weights:        ``(p_chess, p_shogi)`` for ``"weighted"`` strategy.
                        Values are normalised so they need not sum to 1.
        seed:           RNG seed for ``"weighted"`` strategy.
    """

    def __init__(
        self,
        chess_loader: Iterator[ChessBatch],
        shogi_loader: Iterator[ShogiBatch],
        interleave: Literal["round_robin", "weighted"] = "round_robin",
        weights: tuple[float, float] = (0.5, 0.5),
        seed: int = 0,
    ) -> None:
        if interleave not in ("round_robin", "weighted"):
            raise ValueError(
                f"interleave must be 'round_robin' or 'weighted', got {interleave!r}"
            )
        self._chess_loader = chess_loader
        self._shogi_loader = shogi_loader
        self.interleave = interleave
        # Normalise weights to a probability
        total = weights[0] + weights[1]
        if total <= 0.0:
            raise ValueError("weights must be positive")
        self._p_shogi: float = weights[1] / total
        self._seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chess_iter(self) -> Iterator[ChessBatch]:
        """Wrapping iterator over the chess loader (handles StopIteration)."""
        while True:
            for batch in self._chess_loader:
                yield batch
            # If the underlying loader is finite, start over.
            # For infinite generators this branch is never reached.
            break

    def _shogi_iter(self) -> Iterator[ShogiBatch]:
        """Wrapping iterator over the shogi loader (handles StopIteration)."""
        while True:
            for batch in self._shogi_loader:
                yield batch
            break

    # ------------------------------------------------------------------
    # Public iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[JointBatch]:
        chess_it: Iterator[ChessBatch] = iter(self._chess_loader)
        shogi_it: Iterator[ShogiBatch] = iter(self._shogi_loader)

        def _next_chess() -> JointBatch | None:
            try:
                return _chess_batch_to_joint(next(chess_it))
            except StopIteration:
                return None

        def _next_shogi() -> JointBatch | None:
            try:
                return _shogi_batch_to_joint(next(shogi_it))
            except StopIteration:
                return None

        if self.interleave == "round_robin":
            # Strict alternation — chess first, then shogi, repeat.
            # Both iterators are consumed at the same rate.
            # If one is finite and shorter, we stop when it's exhausted.
            chess_turn = True
            while True:
                if chess_turn:
                    item = _next_chess()
                    if item is None:
                        return
                    yield item
                    chess_turn = False
                else:
                    item = _next_shogi()
                    if item is None:
                        return
                    yield item
                    chess_turn = True

        else:  # "weighted"
            rng = random.Random(self._seed)
            while True:
                use_shogi = rng.random() < self._p_shogi
                if use_shogi:
                    item = _next_shogi()
                    if item is None:
                        return
                    yield item
                else:
                    item = _next_chess()
                    if item is None:
                        return
                    yield item
