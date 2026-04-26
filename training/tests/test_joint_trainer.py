"""
Tests for training.trainers.supervised.JointSupervisedTrainer.

Coverage:
  - 5 joint training steps on synthetic chess + shogi without crashing.
  - Per-game loss is tracked separately in history.
  - Checkpoint includes sidecar JSON with game="both".
  - Passing a non-JointBatch raises TypeError.
  - Model with game="both" produces safetensors with both heads.
  - Loss values are finite after 5 steps.
  - LR schedule is applied.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from training.data.chessbench_loader import make_chess_loader
from training.data.dlshogi_loader import make_shogi_loader
from training.data.joint_loader import JointBatch, JointLoader
from training.models.transformer import ChessShogiTransformer
from training.trainers.supervised import JointSupervisedTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
N_LAYERS = 2
D_MODEL = 64
N_HEADS = 4
FFN_DIM = 128


def _tiny_joint_model() -> ChessShogiTransformer:
    return ChessShogiTransformer(
        game="both",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        ffn_dim=FFN_DIM,
    )


def _joint_loader(n_steps: int = 10) -> JointLoader:
    chess_loader = make_chess_loader(
        synthetic=True, batch_size=BATCH_SIZE, seed=0, max_batches=n_steps
    )
    shogi_loader = make_shogi_loader(
        synthetic=True, batch_size=BATCH_SIZE, seed=1, max_batches=n_steps
    )
    return JointLoader(
        chess_loader=iter(chess_loader),
        shogi_loader=iter(shogi_loader),
        interleave="round_robin",
    )


def _run_n_joint_steps(
    n_steps: int = 5,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 0,
) -> tuple[JointSupervisedTrainer, list[dict[str, float]]]:
    model = _tiny_joint_model()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = checkpoint_dir or tmp
        trainer = JointSupervisedTrainer(
            model=model,
            lr=1e-3,
            warmup_steps=1,
            total_steps=n_steps,
            checkpoint_dir=ckpt_dir,
            checkpoint_every=checkpoint_every,
        )
        loader = _joint_loader(n_steps=n_steps)
        metrics = trainer.train(
            data_iter=loader,
            num_steps=n_steps,
            log_every=1,
            verbose=True,
        )
    return trainer, metrics


# ---------------------------------------------------------------------------
# Basic operation tests
# ---------------------------------------------------------------------------


class TestJointTrainerBasic:
    def test_5_steps_no_crash(self) -> None:
        """5 joint training steps must complete without exception."""
        trainer, metrics = _run_n_joint_steps(5)
        assert len(metrics) == 5, f"Expected 5 metrics, got {len(metrics)}"

    def test_5_steps_finite_loss(self) -> None:
        """All loss values after 5 steps must be finite."""
        _, metrics = _run_n_joint_steps(5)
        for m in metrics:
            assert math.isfinite(m["loss"]), f"Non-finite loss at step: {m}"

    def test_5_steps_positive_loss(self) -> None:
        """All loss values must be strictly positive."""
        _, metrics = _run_n_joint_steps(5)
        for m in metrics:
            assert m["loss"] > 0.0, f"Zero or negative loss: {m}"

    def test_step_counter_increments(self) -> None:
        """Trainer.step must equal n_steps after training."""
        trainer, _ = _run_n_joint_steps(5)
        assert trainer.step == 5


# ---------------------------------------------------------------------------
# History / per-game tracking tests
# ---------------------------------------------------------------------------


class TestJointTrainerHistory:
    def test_history_length_matches_steps(self) -> None:
        """trainer.history must have one entry per training step."""
        trainer, _ = _run_n_joint_steps(6)
        assert len(trainer.history) == 6

    def test_per_game_loss_keys_present(self) -> None:
        """Each history entry must have a game-specific loss key."""
        trainer, _ = _run_n_joint_steps(6)
        for entry in trainer.history:
            game_tag = "chess" if entry["game"] == 0.0 else "shogi"
            loss_key = f"loss_{game_tag}"
            assert loss_key in entry, (
                f"Missing key '{loss_key}' in history entry: {entry.keys()}"
            )

    def test_chess_and_shogi_appear_in_history(self) -> None:
        """With round_robin and ≥2 steps, both games must appear in history."""
        trainer, _ = _run_n_joint_steps(6)
        games_seen = {int(e["game"]) for e in trainer.history}
        assert 0 in games_seen, "chess (game=0.0) never appeared in history"
        assert 1 in games_seen, "shogi (game=1.0) never appeared in history"

    def test_alternating_game_sequence_in_history(self) -> None:
        """Round-robin must produce alternating game tags in history."""
        trainer, _ = _run_n_joint_steps(6)
        games = [int(e["game"]) for e in trainer.history]
        expected = [0, 1] * 3  # chess=0, shogi=1 for 6 steps
        assert games == expected, f"Expected {expected}, got {games}"

    def test_lr_recorded_in_history(self) -> None:
        """Each history entry must contain a 'lr' key."""
        trainer, _ = _run_n_joint_steps(4)
        for entry in trainer.history:
            assert "lr" in entry, f"Missing 'lr' in entry: {entry}"

    def test_step_recorded_in_history(self) -> None:
        """Each history entry must record its step index."""
        trainer, _ = _run_n_joint_steps(4)
        steps = [int(e["step"]) for e in trainer.history]
        assert steps == list(range(4)), f"Unexpected step sequence: {steps}"


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestJointTrainerCheckpoint:
    def test_checkpoint_creates_safetensors(self) -> None:
        """Checkpoint must create a .safetensors file."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_joint_model()
            trainer = JointSupervisedTrainer(
                model=model,
                lr=1e-3,
                warmup_steps=1,
                total_steps=4,
                checkpoint_dir=tmp,
                checkpoint_every=0,  # manual only
            )
            loader = _joint_loader(n_steps=2)
            trainer.train(data_iter=loader, num_steps=2, verbose=False)
            saved_path = trainer.save_final()
            assert saved_path.exists(), f"safetensors not found at {saved_path}"
            assert saved_path.suffix == ".safetensors"

    def test_checkpoint_json_has_game_both(self) -> None:
        """Sidecar JSON must record game='both' so the engine can serve both games."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_joint_model()
            trainer = JointSupervisedTrainer(
                model=model,
                lr=1e-3,
                warmup_steps=1,
                total_steps=4,
                checkpoint_dir=tmp,
                checkpoint_every=0,
            )
            loader = _joint_loader(n_steps=2)
            trainer.train(data_iter=loader, num_steps=2, verbose=False)
            saved_path = trainer.save_final()

            json_path = saved_path.with_suffix(".json")
            assert json_path.exists(), f"JSON sidecar not found at {json_path}"
            with open(json_path) as f:
                meta = json.load(f)
            assert meta.get("game") == "both", (
                f"Expected game='both' in sidecar JSON, got {meta.get('game')!r}"
            )

    def test_periodic_checkpoint_created(self) -> None:
        """With checkpoint_every=2 and 4 steps, at least one checkpoint is saved."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_joint_model()
            trainer = JointSupervisedTrainer(
                model=model,
                lr=1e-3,
                warmup_steps=1,
                total_steps=4,
                checkpoint_dir=tmp,
                checkpoint_every=2,  # save at step 2
            )
            loader = _joint_loader(n_steps=4)
            trainer.train(data_iter=loader, num_steps=4, verbose=False)

            safetensors_files = list(Path(tmp).glob("*.safetensors"))
            assert safetensors_files, "No checkpoint files found"


# ---------------------------------------------------------------------------
# Model structure tests
# ---------------------------------------------------------------------------


class TestJointModelStructure:
    def test_model_game_both_required(self) -> None:
        """JointSupervisedTrainer must reject models with game != 'both'."""
        chess_model = ChessShogiTransformer(
            game="chess", n_layers=2, d_model=64, n_heads=4, ffn_dim=128
        )
        with pytest.raises(ValueError, match="game='both'"):
            JointSupervisedTrainer(model=chess_model)

    def test_joint_model_has_both_policy_heads(self) -> None:
        """Model with game='both' must have both chess_policy_head and shogi_policy_head."""
        model = _tiny_joint_model()
        assert hasattr(model, "chess_policy_head"), "Missing chess_policy_head"
        assert hasattr(model, "shogi_policy_head"), "Missing shogi_policy_head"

    def test_chess_forward_correct_shape(self) -> None:
        """Chess forward through joint model must return [B, 1858] policy."""
        model = _tiny_joint_model()
        x = mx.array(
            np.random.default_rng(0).random((BATCH_SIZE, 64, 19)).astype(np.float32)
        )
        out = model(x, "chess")
        mx.eval(out["policy"])
        assert out["policy"].shape == (BATCH_SIZE, 1858)

    def test_shogi_forward_correct_shape(self) -> None:
        """Shogi forward through joint model must return [B, 2187] policy."""
        model = _tiny_joint_model()
        x = mx.array(
            np.random.default_rng(1).random((BATCH_SIZE, 81, 90)).astype(np.float32)
        )
        out = model(x, "shogi")
        mx.eval(out["policy"])
        assert out["policy"].shape == (BATCH_SIZE, 2187)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestJointTrainerErrors:
    def test_non_joint_batch_raises(self) -> None:
        """Passing a non-JointBatch to train() must raise TypeError."""
        model = _tiny_joint_model()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = JointSupervisedTrainer(
                model=model, lr=1e-3, warmup_steps=1, total_steps=1,
                checkpoint_dir=tmp, checkpoint_every=0,
            )
            # Create a fake iterable with a non-JointBatch item
            class _FakeBatch:
                pass

            with pytest.raises(TypeError, match="JointBatch"):
                trainer.train(
                    data_iter=[_FakeBatch()],
                    num_steps=1,
                    verbose=False,
                )
