"""
Loss functions for training ChessShogiTransformer.

Three losses, combined with configurable weights:
    policy_loss:     Cross-entropy over move logits (soft targets supported).
    value_loss:      MSE between predicted value and game outcome.
    moves_left_loss: L2 (MSE) between predicted and actual half-moves remaining.

Combined:
    total = w_policy * policy_loss + w_value * value_loss + w_ml * moves_left_loss
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def policy_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    eps: float = 1e-8,
) -> mx.array:
    """Cross-entropy loss supporting both hard and soft (probability) targets.

    For hard labels (one-hot), this reduces to standard CE.
    For soft labels (probability distributions), this computes the full
    KL-like cross-entropy.

    Args:
        logits:  `[B, num_moves]` unnormalised log-probabilities.
        targets: `[B, num_moves]` probability targets (sum to 1 per sample,
                 or one-hot).
        eps:     Small constant to avoid log(0).

    Returns:
        Scalar mean loss.
    """
    log_probs = logits - mx.log(mx.exp(logits).sum(axis=-1, keepdims=True) + eps)
    # Cross-entropy: -sum(targets * log_probs)
    loss = -(targets * log_probs).sum(axis=-1)  # [B]
    return loss.mean()


def value_mse(
    predicted: mx.array,
    target: mx.array,
) -> mx.array:
    """Mean squared error for value head.

    Args:
        predicted: `[B, 1]` from value head (tanh output).
        target:    `[B, 1]` ground-truth value in [-1, +1].

    Returns:
        Scalar mean loss.
    """
    return ((predicted - target) ** 2).mean()


def moves_left_l2(
    predicted: mx.array,
    target: mx.array,
) -> mx.array:
    """L2 (MSE) loss for moves-left head.

    Args:
        predicted: `[B, 1]` softplus output (≥ 0).
        target:    `[B, 1]` actual half-moves remaining (≥ 0).

    Returns:
        Scalar mean loss.
    """
    return ((predicted - target) ** 2).mean()


def combined_loss(
    policy_logits: mx.array,
    value_pred: mx.array,
    moves_left_pred: mx.array,
    policy_target: mx.array,
    value_target: mx.array,
    moves_left_target: mx.array,
    w_policy: float = 1.0,
    w_value: float = 1.0,
    w_moves_left: float = 0.1,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Compute combined training loss.

    Args:
        policy_logits:     `[B, num_moves]` raw policy logits.
        value_pred:        `[B, 1]` value prediction (tanh).
        moves_left_pred:   `[B, 1]` moves-left prediction (softplus).
        policy_target:     `[B, num_moves]` probability targets.
        value_target:      `[B, 1]` value targets.
        moves_left_target: `[B, 1]` moves-left targets.
        w_policy:          Weight for policy loss (default 1.0).
        w_value:           Weight for value loss (default 1.0).
        w_moves_left:      Weight for moves-left loss (default 0.1).

    Returns:
        Tuple of (total_loss, {component_losses}).
    """
    p_loss = policy_cross_entropy(policy_logits, policy_target)
    v_loss = value_mse(value_pred, value_target)
    ml_loss = moves_left_l2(moves_left_pred, moves_left_target)

    total = w_policy * p_loss + w_value * v_loss + w_moves_left * ml_loss

    return total, {
        "policy": p_loss,
        "value": v_loss,
        "moves_left": ml_loss,
        "total": total,
    }
