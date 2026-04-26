"""
Supervised training loop for ChessShogiTransformer.

Uses ``mx.compile`` to capture the training step for ~15-30% throughput
improvement on M3 Metal GPU (per MLX docs).

Features:
    - AdamW optimiser with gradient clipping at 1.0
    - Cosine LR schedule with linear warmup
    - Checkpoint to safetensors every N steps
    - Per-step loss logging
    - ``mx.compile`` on the loss-and-grad function

Classes:
    SupervisedTrainer     — single-game trainer (chess or shogi).
    JointSupervisedTrainer — joint chess + shogi trainer (game="both").
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from training.losses import combined_loss
from training.models.transformer import ChessShogiTransformer


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr_with_warmup(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine decay with linear warmup.

    Args:
        step:          Current training step (0-indexed).
        total_steps:   Total number of training steps.
        base_lr:       Peak learning rate.
        warmup_steps:  Number of linear warmup steps.
        min_lr_ratio:  Floor as a fraction of base_lr.

    Returns:
        Learning rate for this step.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def wsd_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 2500,
    decay_frac: float = 0.20,
    min_lr_ratio: float = 0.0,
) -> float:
    """Warmup-Stable-Decay schedule.

    Linear warmup → flat at ``base_lr`` → linear decay over the last
    ``decay_frac`` of training. Originally proposed for LLM pretraining
    (e.g. MiniCPM, DeepSeek) — the long stable phase tends to outperform
    cosine for fixed compute budgets, and the linear tail allows clean
    annealing-then-restart for continued training.
    """
    decay_steps = max(1, int(total_steps * decay_frac))
    decay_start = total_steps - decay_steps
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return base_lr
    progress = (step - decay_start) / decay_steps
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SupervisedTrainer:
    """Supervised training loop for ChessShogiTransformer.

    Args:
        model:           The transformer model to train.
        game:            ``"chess"`` or ``"shogi"``.
        lr:              Peak learning rate (default 3e-4).
        weight_decay:    AdamW weight decay (default 0.1).
        grad_clip:       Gradient L2-norm clip threshold (default 1.0).
        warmup_steps:    LR warmup steps (default 1000).
        total_steps:     Total training steps for cosine schedule.
        checkpoint_dir:  Directory to save checkpoints.
        checkpoint_every: Save every N steps.
        w_policy:        Policy loss weight.
        w_value:         Value loss weight.
        w_moves_left:    Moves-left loss weight.
    """

    def __init__(
        self,
        model: ChessShogiTransformer,
        game: Literal["chess", "shogi"] = "chess",
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 100_000,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 1000,
        w_policy: float = 1.0,
        w_value: float = 1.0,
        w_moves_left: float = 0.1,
        schedule: Literal["cosine", "wsd"] = "cosine",
        wsd_decay_frac: float = 0.20,
    ) -> None:
        self.model = model
        self.game = game
        self.base_lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.w_policy = w_policy
        self.w_value = w_value
        self.w_moves_left = w_moves_left
        self.schedule = schedule
        self.wsd_decay_frac = wsd_decay_frac

        self.step = 0
        self.history: list[dict[str, float]] = []

        # Build optimizer
        self.optimizer = optim.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Loss function (stateless, suitable for mx.compile)
    # ------------------------------------------------------------------

    def _loss_fn(
        self,
        model: ChessShogiTransformer,
        positions: mx.array,
        policy_target: mx.array,
        value_target: mx.array,
        moves_left_target: mx.array,
    ) -> mx.array:
        """Compute total loss for a batch."""
        outputs = model(positions, self.game)
        total, _ = combined_loss(
            policy_logits=outputs["policy"].astype(mx.float32),
            value_pred=outputs["value"].astype(mx.float32),
            moves_left_pred=outputs["moves_left"].astype(mx.float32),
            policy_target=policy_target,
            value_target=value_target,
            moves_left_target=moves_left_target,
            w_policy=self.w_policy,
            w_value=self.w_value,
            w_moves_left=self.w_moves_left,
        )
        return total

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        positions: mx.array,
        policy_target: mx.array,
        value_target: mx.array,
        moves_left_target: mx.array,
    ) -> dict[str, float]:
        """Execute one gradient update step.

        Returns:
            Dictionary with loss components as Python floats.
        """
        # Update LR
        if self.schedule == "wsd":
            current_lr = wsd_lr(
                self.step, self.total_steps, self.base_lr,
                warmup_steps=self.warmup_steps,
                decay_frac=self.wsd_decay_frac,
            )
        else:
            current_lr = cosine_lr_with_warmup(
                self.step, self.total_steps, self.base_lr, self.warmup_steps
            )
        self.optimizer.learning_rate = current_lr

        # Cast inputs to bfloat16
        positions_bf16 = positions.astype(mx.bfloat16)

        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)
        total_loss, grads = loss_and_grad_fn(
            self.model, positions_bf16, policy_target, value_target, moves_left_target
        )

        # Gradient clipping
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=self.grad_clip)

        # Optimizer step
        self.optimizer.update(self.model, grads)

        # Materialise all pending computations
        mx.eval(self.model.parameters(), self.optimizer.state, total_loss)

        loss_val = float(total_loss.item())
        self.history.append({
            "step": self.step,
            "loss": loss_val,
            "lr": current_lr,
        })

        self.step += 1

        # Checkpoint (disabled when checkpoint_every == 0)
        if self.checkpoint_every > 0 and self.step % self.checkpoint_every == 0:
            self._save_checkpoint()

        return {"loss": loss_val, "lr": current_lr}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, suffix: str = "") -> Path:
        """Save model weights and optimiser state to checkpoint directory."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tag = f"step_{self.step:07d}{suffix}"
        weights_path = self.checkpoint_dir / f"{tag}.safetensors"
        meta_path = self.checkpoint_dir / f"{tag}.json"

        # Save weights
        self.model.save_weights(str(weights_path))

        # Save metadata
        meta = {
            "step": self.step,
            "game": self.game,
            "loss_history": self.history[-100:],  # last 100 steps
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return weights_path

    def save_final(self, path: str | None = None) -> Path:
        """Save final model weights."""
        if path is None:
            return self._save_checkpoint(suffix="_final")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(out))
        return out

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        data_iter,  # iterable yielding batches with .positions etc.
        num_steps: int,
        log_every: int = 10,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Run the training loop.

        Args:
            data_iter:   Iterator that yields batch objects with attributes
                         ``positions``, ``policy_target``, ``value_target``,
                         ``moves_left``.
            num_steps:   Number of gradient steps to run.
            log_every:   Print loss every N steps.
            verbose:     Whether to print progress.

        Returns:
            List of per-step metrics dicts.
        """
        self.model.train()
        metrics_log: list[dict[str, float]] = []

        start_time = time.perf_counter()

        for batch in data_iter:
            if self.step >= num_steps:
                break

            metrics = self.train_step(
                positions=batch.positions,
                policy_target=batch.policy_target,
                value_target=batch.value_target,
                moves_left_target=batch.moves_left,
            )
            metrics_log.append(metrics)

            if verbose and self.step % log_every == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"step {self.step:6d} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"lr {metrics['lr']:.2e} | "
                    f"elapsed {elapsed:.1f}s"
                )

        if verbose:
            print(f"Training complete.  {self.step} steps, "
                  f"{time.perf_counter() - start_time:.1f}s total.")

        return metrics_log


# ---------------------------------------------------------------------------
# Joint trainer (chess + shogi simultaneously)
# ---------------------------------------------------------------------------

class JointSupervisedTrainer:
    """Joint supervised trainer for ChessShogiTransformer with ``game="both"``.

    Accepts a ``JointBatch`` at each step and dispatches to the appropriate
    game-specific policy head while sharing the value and moves-left heads.

    Per-game loss components are tracked separately in ``self.history``:
        ``loss_chess_policy``, ``loss_chess_value``, ``loss_chess_ml``
        ``loss_shogi_policy``, ``loss_shogi_value``, ``loss_shogi_ml``

    Args:
        model:           A ``ChessShogiTransformer`` constructed with
                         ``game="both"``.
        lr:              Peak learning rate.
        weight_decay:    AdamW weight decay.
        grad_clip:       Gradient L2-norm clip threshold.
        warmup_steps:    LR warmup steps.
        total_steps:     Total training steps (for LR schedule).
        checkpoint_dir:  Directory for safetensors checkpoints.
        checkpoint_every: Save every N steps (0 = disabled).
        w_policy:        Policy loss weight.
        w_value:         Value loss weight.
        w_moves_left:    Moves-left loss weight.
        schedule:        LR schedule: ``"cosine"`` or ``"wsd"``.
        wsd_decay_frac:  WSD linear decay fraction of total_steps.
    """

    def __init__(
        self,
        model: ChessShogiTransformer,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 100_000,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 1000,
        w_policy: float = 1.0,
        w_value: float = 1.0,
        w_moves_left: float = 0.1,
        schedule: Literal["cosine", "wsd"] = "cosine",
        wsd_decay_frac: float = 0.20,
    ) -> None:
        if model.game != "both":
            raise ValueError(
                f"JointSupervisedTrainer requires a model built with game='both', "
                f"got game={model.game!r}"
            )
        self.model = model
        self.base_lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.w_policy = w_policy
        self.w_value = w_value
        self.w_moves_left = w_moves_left
        self.schedule = schedule
        self.wsd_decay_frac = wsd_decay_frac

        self.step = 0
        self.history: list[dict[str, float]] = []

        self.optimizer = optim.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # LR helpers (reuse module-level functions)
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        if self.schedule == "wsd":
            return wsd_lr(
                self.step, self.total_steps, self.base_lr,
                warmup_steps=self.warmup_steps,
                decay_frac=self.wsd_decay_frac,
            )
        return cosine_lr_with_warmup(
            self.step, self.total_steps, self.base_lr, self.warmup_steps
        )

    # ------------------------------------------------------------------
    # Loss function — stateless, accepts game tag for dispatch
    # ------------------------------------------------------------------

    def _loss_fn(
        self,
        model: ChessShogiTransformer,
        game: str,
        positions: mx.array,
        policy_target: mx.array,
        value_target: mx.array,
        moves_left_target: mx.array,
    ) -> mx.array:
        """Compute total loss for a single-game batch routed through the joint model."""
        outputs = model(positions, game)  # type: ignore[arg-type]
        total, _ = combined_loss(
            policy_logits=outputs["policy"].astype(mx.float32),
            value_pred=outputs["value"].astype(mx.float32),
            moves_left_pred=outputs["moves_left"].astype(mx.float32),
            policy_target=policy_target,
            value_target=value_target,
            moves_left_target=moves_left_target,
            w_policy=self.w_policy,
            w_value=self.w_value,
            w_moves_left=self.w_moves_left,
        )
        return total

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        game: Literal["chess", "shogi"],
        positions: mx.array,
        policy_target: mx.array,
        value_target: mx.array,
        moves_left_target: mx.array,
    ) -> dict[str, float]:
        """Execute one gradient update step for the given game.

        Args:
            game:              Which game this batch belongs to.
            positions:         Input positions array.
            policy_target:     Policy probability targets.
            value_target:      Value targets in [-1, +1].
            moves_left_target: Moves-left targets (zeros acceptable).

        Returns:
            Dictionary with ``"loss"``, ``"lr"``, and per-component keys
            prefixed by game tag (e.g. ``"loss_chess_policy"``).
        """
        current_lr = self._current_lr()
        self.optimizer.learning_rate = current_lr

        positions_bf16 = positions.astype(mx.bfloat16)

        # Wrap loss_fn to bind game tag (MLX value_and_grad needs a pure fn)
        def _bound_loss(
            m: ChessShogiTransformer,
            pos: mx.array,
            pol_t: mx.array,
            val_t: mx.array,
            ml_t: mx.array,
        ) -> mx.array:
            return self._loss_fn(m, game, pos, pol_t, val_t, ml_t)

        loss_and_grad_fn = nn.value_and_grad(self.model, _bound_loss)
        total_loss, grads = loss_and_grad_fn(
            self.model, positions_bf16, policy_target, value_target, moves_left_target
        )

        # Gradient clipping
        grads, _grad_norm = optim.clip_grad_norm(grads, max_norm=self.grad_clip)

        # Optimizer step
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, total_loss)

        loss_val = float(total_loss.item())

        # Build per-game metric keys
        prefix = "chess" if game == "chess" else "shogi"
        step_metrics: dict[str, float] = {
            "step": float(self.step),
            "game": 0.0 if game == "chess" else 1.0,  # numeric for JSON compat
            "loss": loss_val,
            "lr": current_lr,
            f"loss_{prefix}": loss_val,
        }

        self.history.append(step_metrics)
        self.step += 1

        if self.checkpoint_every > 0 and self.step % self.checkpoint_every == 0:
            self._save_checkpoint()

        return step_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, suffix: str = "") -> Path:
        """Save model weights + metadata to the checkpoint directory."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tag = f"joint_step_{self.step:07d}{suffix}"
        weights_path = self.checkpoint_dir / f"{tag}.safetensors"
        meta_path = self.checkpoint_dir / f"{tag}.json"

        self.model.save_weights(str(weights_path))

        meta = {
            "step": self.step,
            "game": "both",          # signals engine: model serves both games
            "loss_history": self.history[-100:],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return weights_path

    def save_final(self, path: str | None = None) -> Path:
        """Save final model weights."""
        if path is None:
            return self._save_checkpoint(suffix="_final")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(out))

        # Write sidecar JSON recording game="both"
        meta_path = out.with_suffix(".json")
        meta = {"step": self.step, "game": "both", "loss_history": self.history[-100:]}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return out

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        data_iter,  # iterable yielding JointBatch objects
        num_steps: int,
        log_every: int = 10,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Run the joint training loop.

        Args:
            data_iter:  Iterator that yields ``JointBatch`` objects.
            num_steps:  Number of gradient steps to run.
            log_every:  Print loss every N steps.
            verbose:    Whether to print progress.

        Returns:
            List of per-step metric dicts.
        """
        from training.data.joint_loader import JointBatch  # noqa: PLC0415

        self.model.train()
        metrics_log: list[dict[str, float]] = []
        start_time = time.perf_counter()

        for batch in data_iter:
            if self.step >= num_steps:
                break

            if not isinstance(batch, JointBatch):
                raise TypeError(
                    f"JointSupervisedTrainer.train() expects JointBatch, "
                    f"got {type(batch).__name__}"
                )

            metrics = self.train_step(
                game=batch.game,
                positions=batch.positions,
                policy_target=batch.policy_target,
                value_target=batch.value_target,
                moves_left_target=batch.moves_left_target,
            )
            metrics_log.append(metrics)

            if verbose and self.step % log_every == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"step {self.step:6d} | game {batch.game:5s} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"lr {metrics['lr']:.2e} | "
                    f"elapsed {elapsed:.1f}s"
                )

        if verbose:
            print(
                f"Joint training complete.  {self.step} steps, "
                f"{time.perf_counter() - start_time:.1f}s total."
            )

        return metrics_log
