"""
Tests for ChessShogiTransformer.

Verifies:
1. Forward pass on chess batch of size 2 → correct output shapes.
2. Backward pass (gradient computation) doesn't produce NaN.
3. Parameter count within ±10% of 40M target.
4. Value head output is in [-1, +1].
5. Moves-left head output is non-negative.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from training.models.transformer import (
    CHESS_NUM_MOVES,
    SHOGI_NUM_MOVES,
    ChessShogiTransformer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chess_model() -> ChessShogiTransformer:
    """Small chess model for fast tests."""
    return ChessShogiTransformer(game="chess")


@pytest.fixture(scope="module")
def chess_batch() -> mx.array:
    """Fake chess input batch [2, 64, 19]."""
    np_arr = np.random.default_rng(0).random((2, 64, 19)).astype(np.float32)
    return mx.array(np_arr)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_chess_forward_shapes(chess_model: ChessShogiTransformer, chess_batch: mx.array) -> None:
    """Forward pass on chess batch of size 2 produces correct shapes."""
    chess_model.eval()
    outputs = chess_model(chess_batch, game="chess")
    mx.eval(outputs["policy"], outputs["value"], outputs["moves_left"])

    policy = outputs["policy"]
    value = outputs["value"]
    moves_left = outputs["moves_left"]

    assert policy.shape == (2, CHESS_NUM_MOVES), (
        f"Policy shape {policy.shape} != (2, {CHESS_NUM_MOVES})"
    )
    assert value.shape == (2, 1), f"Value shape {value.shape} != (2, 1)"
    assert moves_left.shape == (2, 1), f"Moves-left shape {moves_left.shape} != (2, 1)"


def test_value_range(chess_model: ChessShogiTransformer, chess_batch: mx.array) -> None:
    """Value head output must be in [-1, +1] (tanh)."""
    chess_model.eval()
    outputs = chess_model(chess_batch, game="chess")
    v = np.array(outputs["value"].astype(mx.float32))
    mx.eval(outputs["value"])
    v = np.array(outputs["value"].astype(mx.float32))
    assert np.all(v >= -1.0 - 1e-5) and np.all(v <= 1.0 + 1e-5), (
        f"Value out of range: min={v.min():.4f}, max={v.max():.4f}"
    )


def test_moves_left_nonnegative(chess_model: ChessShogiTransformer, chess_batch: mx.array) -> None:
    """Moves-left head output must be non-negative (softplus)."""
    chess_model.eval()
    outputs = chess_model(chess_batch, game="chess")
    mx.eval(outputs["moves_left"])
    ml = np.array(outputs["moves_left"].astype(mx.float32))
    assert np.all(ml >= 0.0 - 1e-5), f"Moves-left negative: min={ml.min():.4f}"


# ---------------------------------------------------------------------------
# Backward pass — no NaN
# ---------------------------------------------------------------------------


def test_backward_no_nan() -> None:
    """Gradient computation must not produce NaN."""
    model = ChessShogiTransformer(game="chess")
    model.train()

    x = mx.array(np.random.default_rng(42).random((2, 64, 19)).astype(np.float32))
    policy_target = np.zeros((2, CHESS_NUM_MOVES), dtype=np.float32)
    policy_target[0, 0] = 1.0
    policy_target[1, 100] = 1.0
    pt = mx.array(policy_target)
    vt = mx.array(np.array([[0.5], [-0.3]], dtype=np.float32))
    mlt = mx.array(np.array([[40.0], [20.0]], dtype=np.float32))

    def loss_fn(model: ChessShogiTransformer) -> mx.array:
        from training.losses import combined_loss  # noqa: PLC0415

        out = model(x, "chess")
        total, _ = combined_loss(
            out["policy"].astype(mx.float32),
            out["value"].astype(mx.float32),
            out["moves_left"].astype(mx.float32),
            pt, vt, mlt,
        )
        return total

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_grad_fn(model)
    mx.eval(loss, *[g for _, g in grads.items() if isinstance(g, mx.array)])

    loss_val = float(loss.item())
    assert math.isfinite(loss_val), f"Loss is non-finite: {loss_val}"
    assert not math.isnan(loss_val), f"Loss is NaN: {loss_val}"

    # Check no NaN in gradients (flatten the nested grad dict)
    def _check_no_nan(d: dict | mx.array, path: str = "") -> None:
        if isinstance(d, mx.array):
            mx.eval(d)
            arr = np.array(d.astype(mx.float32))
            assert not np.any(np.isnan(arr)), f"NaN gradient at {path}: {arr}"
        elif isinstance(d, dict):
            for k, v in d.items():
                _check_no_nan(v, f"{path}.{k}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                _check_no_nan(v, f"{path}[{i}]")

    _check_no_nan(grads)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def test_parameter_count() -> None:
    """Parameter count must be within ±10% of 40M target."""
    model = ChessShogiTransformer(game="chess")
    count = model.count_parameters()
    target = 40_000_000
    tolerance = 0.10  # ±10%

    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)

    print(f"\nParameter count: {count:,}  (target: {target:,}, "
          f"range: {lower:,.0f}-{upper:,.0f})")
    assert lower <= count <= upper, (
        f"Parameter count {count:,} outside ±10% of {target:,} "
        f"(range: {lower:,.0f}-{upper:,.0f})"
    )


# ---------------------------------------------------------------------------
# Shogi forward (if model is instantiated for both)
# ---------------------------------------------------------------------------


def test_shogi_forward_shapes() -> None:
    """Forward pass on shogi batch of size 2 produces correct shapes."""
    model = ChessShogiTransformer(game="shogi")
    model.eval()

    x = mx.array(np.random.default_rng(1).random((2, 81, 90)).astype(np.float32))
    outputs = model(x, game="shogi")
    mx.eval(outputs["policy"], outputs["value"], outputs["moves_left"])

    assert outputs["policy"].shape == (2, SHOGI_NUM_MOVES)
    assert outputs["value"].shape == (2, 1)
    assert outputs["moves_left"].shape == (2, 1)


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


def test_forward_deterministic(chess_model: ChessShogiTransformer, chess_batch: mx.array) -> None:
    """Forward pass must be deterministic (same input → same output)."""
    chess_model.eval()
    out1 = chess_model(chess_batch, "chess")
    out2 = chess_model(chess_batch, "chess")
    mx.eval(out1["policy"], out2["policy"])

    p1 = np.array(out1["policy"].astype(mx.float32))
    p2 = np.array(out2["policy"].astype(mx.float32))
    np.testing.assert_array_equal(p1, p2, err_msg="Forward pass is not deterministic")
