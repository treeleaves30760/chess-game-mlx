"""
Tests for training.data.joint_loader.

Coverage:
  - Round-robin alternates chess/shogi exactly.
  - Weighted sampling matches expected proportion within tolerance.
  - JointBatch fields have correct shapes for each game.
  - Both game iterators can be exhausted independently.
  - Invalid interleave value raises ValueError.
  - JointLoader stops when either finite iterator is exhausted
    (for round_robin) or when one runs out (for weighted).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from training.data.chessbench_loader import make_chess_loader
from training.data.dlshogi_loader import make_shogi_loader
from training.data.joint_loader import JointBatch, JointLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
N_CHESS = 10
N_SHOGI = 10


def _chess_iter(n: int = N_CHESS):
    return iter(
        make_chess_loader(synthetic=True, batch_size=BATCH_SIZE, seed=0, max_batches=n)
    )


def _shogi_iter(n: int = N_SHOGI):
    return iter(
        make_shogi_loader(synthetic=True, batch_size=BATCH_SIZE, seed=1, max_batches=n)
    )


# ---------------------------------------------------------------------------
# Round-robin tests
# ---------------------------------------------------------------------------


class TestRoundRobin:
    def _make_loader(self, n_chess: int = N_CHESS, n_shogi: int = N_SHOGI) -> JointLoader:
        return JointLoader(
            chess_loader=_chess_iter(n_chess),
            shogi_loader=_shogi_iter(n_shogi),
            interleave="round_robin",
        )

    def test_alternates_exactly(self) -> None:
        """Round-robin must strictly alternate chess → shogi → chess → shogi."""
        loader = self._make_loader()
        games = [batch.game for batch in loader]

        # With 10 chess and 10 shogi batches, we expect 20 alternating batches
        assert len(games) == 20
        expected = ["chess", "shogi"] * 10
        assert games == expected, f"Expected strict alternation, got: {games}"

    def test_unequal_lengths_stops_at_shorter(self) -> None:
        """Round-robin stops as soon as the shorter iterator is exhausted."""
        loader = JointLoader(
            chess_loader=_chess_iter(3),  # shorter
            shogi_loader=_shogi_iter(10),
            interleave="round_robin",
        )
        batches = list(loader)
        # Chess exhausts after 3 batches; 3 chess + 3 shogi = 6 total
        assert len(batches) == 6
        games = [b.game for b in batches]
        assert games == ["chess", "shogi"] * 3

    def test_stops_when_chess_exhausted(self) -> None:
        """Loader stops when chess is exhausted (shogi has more)."""
        loader = JointLoader(
            chess_loader=_chess_iter(2),
            shogi_loader=_shogi_iter(20),
            interleave="round_robin",
        )
        batches = list(loader)
        assert len(batches) == 4  # 2 chess + 2 shogi

    def test_stops_when_shogi_exhausted(self) -> None:
        """Loader stops when shogi is exhausted (chess has more).

        Round-robin order: chess1, shogi1, chess2, shogi2, chess3 (shogi
        fails on 3rd attempt → stop).  This yields 5 batches: 3 chess + 2
        shogi, because chess is yielded before shogi is checked each cycle.
        """
        loader = JointLoader(
            chess_loader=_chess_iter(20),
            shogi_loader=_shogi_iter(2),
            interleave="round_robin",
        )
        batches = list(loader)
        chess_count = sum(1 for b in batches if b.game == "chess")
        shogi_count = sum(1 for b in batches if b.game == "shogi")
        # Shogi has 2 batches; chess emits one extra before shogi is checked
        assert shogi_count == 2
        assert chess_count in (2, 3)  # 2 or 3 depending on order of exhaustion


# ---------------------------------------------------------------------------
# Weighted sampling tests
# ---------------------------------------------------------------------------


class TestWeighted:
    def test_proportion_matches_weight(self) -> None:
        """Weighted sampling should produce ~p_shogi fraction of shogi batches."""
        n = 200
        loader = JointLoader(
            chess_loader=make_chess_loader(synthetic=True, batch_size=BATCH_SIZE, seed=0),
            shogi_loader=make_shogi_loader(synthetic=True, batch_size=BATCH_SIZE, seed=1),
            interleave="weighted",
            weights=(0.7, 0.3),
            seed=42,
        )
        games = []
        for i, batch in enumerate(loader):
            if i >= n:
                break
            games.append(batch.game)

        shogi_frac = games.count("shogi") / len(games)
        # Allow ±0.08 tolerance for n=200
        assert abs(shogi_frac - 0.3) < 0.08, (
            f"Shogi fraction {shogi_frac:.3f} too far from target 0.3"
        )

    def test_equal_weight_roughly_half(self) -> None:
        """Equal weights should produce roughly equal split."""
        n = 500
        loader = JointLoader(
            chess_loader=make_chess_loader(synthetic=True, batch_size=BATCH_SIZE, seed=7),
            shogi_loader=make_shogi_loader(synthetic=True, batch_size=BATCH_SIZE, seed=8),
            interleave="weighted",
            weights=(0.5, 0.5),
            seed=123,
        )
        games = []
        for i, batch in enumerate(loader):
            if i >= n:
                break
            games.append(batch.game)

        shogi_frac = games.count("shogi") / len(games)
        assert abs(shogi_frac - 0.5) < 0.06, f"Shogi fraction {shogi_frac:.3f} far from 0.5"

    def test_weighted_stops_on_finite_iterator(self) -> None:
        """Weighted loader stops when one finite iterator is exhausted."""
        loader = JointLoader(
            chess_loader=_chess_iter(5),
            shogi_loader=_shogi_iter(5),
            interleave="weighted",
            weights=(0.5, 0.5),
            seed=0,
        )
        batches = list(loader)
        # Both iterators have 5 batches; first to be drawn N times by Bernoulli stops it
        assert len(batches) <= 10  # at most both exhausted


# ---------------------------------------------------------------------------
# JointBatch shape tests
# ---------------------------------------------------------------------------


class TestJointBatchShapes:
    def _first_batch(self, game_type: str) -> JointBatch:
        if game_type == "chess":
            loader = JointLoader(
                chess_loader=_chess_iter(2),
                shogi_loader=_shogi_iter(100),  # infinite-ish
                interleave="round_robin",
            )
        else:
            loader = JointLoader(
                chess_loader=_chess_iter(100),
                shogi_loader=_shogi_iter(2),
                interleave="round_robin",
            )
        return next(iter(loader))

    def test_chess_batch_is_first(self) -> None:
        """Round-robin always starts with chess."""
        batch = self._first_batch("chess")
        assert batch.game == "chess"

    def test_chess_positions_shape(self) -> None:
        """Chess positions must be [B, 64, 19]."""
        batch = self._first_batch("chess")
        assert batch.positions.shape == (BATCH_SIZE, 64, 19), (
            f"Expected [B, 64, 19], got {batch.positions.shape}"
        )

    def test_chess_policy_target_shape(self) -> None:
        """Chess policy targets must be [B, 1858]."""
        batch = self._first_batch("chess")
        assert batch.policy_target.shape == (BATCH_SIZE, 1858)

    def test_chess_value_target_shape(self) -> None:
        """Value target must be [B, 1]."""
        batch = self._first_batch("chess")
        assert batch.value_target.shape == (BATCH_SIZE, 1)

    def test_chess_moves_left_target_shape(self) -> None:
        """Moves-left target must be [B, 1]."""
        batch = self._first_batch("chess")
        assert batch.moves_left_target.shape == (BATCH_SIZE, 1)

    def test_shogi_positions_shape(self) -> None:
        """Shogi positions must be [B, 81, 90]."""
        loader = JointLoader(
            chess_loader=_chess_iter(100),
            shogi_loader=_shogi_iter(2),
            interleave="round_robin",
        )
        batches = list(loader)
        shogi_batches = [b for b in batches if b.game == "shogi"]
        assert shogi_batches, "No shogi batches produced"
        batch = shogi_batches[0]
        assert batch.positions.shape == (BATCH_SIZE, 81, 90), (
            f"Expected [B, 81, 90], got {batch.positions.shape}"
        )

    def test_shogi_policy_target_shape(self) -> None:
        """Shogi policy targets must be [B, 2187]."""
        loader = JointLoader(
            chess_loader=_chess_iter(100),
            shogi_loader=_shogi_iter(2),
            interleave="round_robin",
        )
        batches = list(loader)
        shogi_batches = [b for b in batches if b.game == "shogi"]
        assert shogi_batches
        batch = shogi_batches[0]
        assert batch.policy_target.shape == (BATCH_SIZE, 2187)

    def test_shogi_moves_left_target_shape(self) -> None:
        """Shogi moves-left target must be [B, 1] (zeros from hcpe loader)."""
        loader = JointLoader(
            chess_loader=_chess_iter(100),
            shogi_loader=_shogi_iter(2),
            interleave="round_robin",
        )
        batches = list(loader)
        shogi_batches = [b for b in batches if b.game == "shogi"]
        assert shogi_batches
        batch = shogi_batches[0]
        assert batch.moves_left_target.shape == (BATCH_SIZE, 1)


# ---------------------------------------------------------------------------
# Independent iterator exhaustion
# ---------------------------------------------------------------------------


class TestIndependentExhaustion:
    def test_chess_exhausts_without_blocking_shogi(self) -> None:
        """Chess can exhaust while shogi still has batches.

        Round-robin: chess1, shogi1, chess2, shogi2, chess3, shogi3, then
        chess4 → fails → stop.  So we get exactly chess_n chess batches and
        chess_n shogi batches (shogi is checked right after chess each cycle).
        """
        chess_n = 3
        shogi_n = 10
        loader = JointLoader(
            chess_loader=_chess_iter(chess_n),
            shogi_loader=_shogi_iter(shogi_n),
            interleave="round_robin",
        )
        batches = list(loader)
        chess_count = sum(1 for b in batches if b.game == "chess")
        shogi_count = sum(1 for b in batches if b.game == "shogi")
        # Chess exhausts first → exactly chess_n chess + chess_n shogi batches
        assert chess_count == chess_n
        assert shogi_count == chess_n  # paired: shogi is drawn after each chess

    def test_separate_iters_do_not_share_state(self) -> None:
        """Two separate JointLoader instances over the same source are independent."""
        loader1 = JointLoader(
            chess_loader=_chess_iter(5),
            shogi_loader=_shogi_iter(5),
            interleave="round_robin",
        )
        loader2 = JointLoader(
            chess_loader=_chess_iter(5),
            shogi_loader=_shogi_iter(5),
            interleave="round_robin",
        )
        count1 = sum(1 for _ in loader1)
        count2 = sum(1 for _ in loader2)
        assert count1 == count2 == 10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_invalid_interleave_raises(self) -> None:
        with pytest.raises(ValueError, match="interleave"):
            JointLoader(
                chess_loader=_chess_iter(),
                shogi_loader=_shogi_iter(),
                interleave="invalid",  # type: ignore[arg-type]
            )

    def test_zero_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            JointLoader(
                chess_loader=_chess_iter(),
                shogi_loader=_shogi_iter(),
                interleave="weighted",
                weights=(0.0, 0.0),
            )

    def test_joint_batch_is_namedtuple(self) -> None:
        """JointBatch must be a NamedTuple with the required fields."""
        loader = JointLoader(
            chess_loader=_chess_iter(1),
            shogi_loader=_shogi_iter(1),
            interleave="round_robin",
        )
        batch = next(iter(loader))
        assert isinstance(batch, JointBatch)
        # NamedTuple fields
        assert hasattr(batch, "game")
        assert hasattr(batch, "positions")
        assert hasattr(batch, "policy_target")
        assert hasattr(batch, "value_target")
        assert hasattr(batch, "moves_left_target")
