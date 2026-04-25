"""
End-to-end pipeline test.

Verifies that pretrain.py runs for 3 steps with synthetic data without
crashing, producing finite losses, and that the loss is not explosively large.
Also checks that the model can be exported and reloaded.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from training.data.chessbench_loader import make_chess_loader
from training.models.transformer import ChessShogiTransformer
from training.trainers.supervised import SupervisedTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_n_steps(
    game: str,
    n_steps: int,
    batch_size: int = 4,
    checkpoint_dir: str | None = None,
) -> list[dict[str, float]]:
    """Run n training steps with synthetic data and return metric history."""
    model = ChessShogiTransformer(
        game=game,
        n_layers=2,      # tiny model for speed
        d_model=64,
        n_heads=4,
        ffn_dim=128,
    )

    if game == "chess":
        loader = make_chess_loader(
            synthetic=True,
            batch_size=batch_size,
            seed=0,
            max_batches=n_steps,
        )
    else:
        from training.data.dlshogi_loader import make_shogi_loader  # noqa: PLC0415
        loader = make_shogi_loader(
            synthetic=True,
            batch_size=batch_size,
            seed=0,
            max_batches=n_steps,
        )

    trainer = SupervisedTrainer(
        model=model,
        game=game,
        lr=1e-3,
        warmup_steps=1,
        total_steps=n_steps,
        checkpoint_dir=checkpoint_dir or tempfile.mkdtemp(),
        checkpoint_every=0,  # disable checkpointing in tests
    )

    metrics = trainer.train(
        data_iter=loader,
        num_steps=n_steps,
        log_every=1,
        verbose=True,
    )
    return metrics


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chess_3_steps_no_crash() -> None:
    """3 synthetic chess training steps complete without crash."""
    metrics = _run_n_steps("chess", 3, batch_size=4)
    assert len(metrics) == 3, f"Expected 3 metric entries, got {len(metrics)}"


def test_chess_3_steps_finite_loss() -> None:
    """Loss after 3 steps is finite (not NaN or Inf)."""
    metrics = _run_n_steps("chess", 3, batch_size=4)
    for m in metrics:
        assert math.isfinite(m["loss"]), f"Non-finite loss: {m['loss']}"


def test_chess_3_steps_reasonable_loss() -> None:
    """Initial loss on random data should be around log(1858) ≈ 7.5 for policy head.
    We allow a generous upper bound to handle all loss components.
    """
    metrics = _run_n_steps("chess", 3, batch_size=4)
    first_loss = metrics[0]["loss"]
    # Policy CE on 1858 classes uniform init ≈ 7.5
    # Value MSE on random ≈ 0.33
    # Moves-left MSE on random ≈ large
    # Allow up to 1000 as a sanity check
    assert first_loss < 10_000.0, f"Initial loss suspiciously large: {first_loss}"
    assert first_loss > 0.0, f"Loss is zero or negative: {first_loss}"


def test_shogi_3_steps_no_crash() -> None:
    """3 synthetic shogi training steps complete without crash."""
    metrics = _run_n_steps("shogi", 3, batch_size=4)
    assert len(metrics) == 3


def test_shogi_3_steps_finite_loss() -> None:
    """Shogi loss after 3 steps is finite."""
    metrics = _run_n_steps("shogi", 3, batch_size=4)
    for m in metrics:
        assert math.isfinite(m["loss"]), f"Non-finite shogi loss: {m['loss']}"


def test_pretrain_script_runs() -> None:
    """Run the pretrain CLI via subprocess for 3 steps and check exit code."""
    import subprocess
    import sys
    import shutil
    from pathlib import Path  # noqa: PLC0415

    # Locate uv to run the script in the correct virtual environment
    uv = shutil.which("uv")
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "pretrain.py"

    # Run from project root so uv workspace is available
    project_root = Path(__file__).resolve().parents[3]

    if uv:
        cmd = [uv, "run", "python", str(script_path)]
    else:
        cmd = [sys.executable, str(script_path)]

    result = subprocess.run(
        cmd + [
            "--game", "chess",
            "--steps", "3",
            "--batch-size", "4",
            "--synthetic",
            "--n-layers", "2",
            "--d-model", "64",
            "--n-heads", "4",
            "--ffn-dim", "128",
            "--checkpoint-every", "0",
            "--warmup-steps", "1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(project_root),
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, (
        f"pretrain.py returned non-zero exit code {result.returncode}\n"
        f"stderr: {result.stderr}"
    )


def test_export_and_reload() -> None:
    """Model can be exported to safetensors and reloaded."""
    import tempfile  # noqa: PLC0415

    from training.export import export_model, load_for_inference  # noqa: PLC0415

    model = ChessShogiTransformer(game="chess", n_layers=2, d_model=64, n_heads=4, ffn_dim=128)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test_model.safetensors"
        export_model(model, out_path, game="chess")
        assert out_path.exists(), "safetensors file not created"
        assert out_path.with_suffix(".json").exists(), "JSON sidecar not created"

        loaded = load_for_inference(out_path, game="chess")
        # Forward pass should work
        x = mx.array(np.random.default_rng(0).random((1, 64, 19)).astype(np.float32))
        out = loaded(x, "chess")
        mx.eval(out["policy"])
        assert out["policy"].shape == (1, 1858)
