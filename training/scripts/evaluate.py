"""
Evaluation script — run model on fixed test positions and report metrics.

Metrics:
    - Policy top-1 accuracy (legal move with highest logit matches target)
    - Value MSE

Usage::

    uv run python -m training.scripts.evaluate \\
        --checkpoint checkpoints/chess_final.safetensors \\
        --game chess \\
        --num-positions 100 \\
        --synthetic
"""

from __future__ import annotations

import sys

import click
import mlx.core as mx
import numpy as np


@click.command("evaluate")
@click.option("--checkpoint", type=str, required=True, help="Path to safetensors checkpoint.")
@click.option("--game", type=click.Choice(["chess", "shogi"]), default="chess")
@click.option("--num-positions", type=int, default=100, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--synthetic", is_flag=True, default=True)
@click.option("--data-dir", type=str, default=None)
@click.option("--seed", type=int, default=0)
def main(
    checkpoint: str,
    game: str,
    num_positions: int,
    batch_size: int,
    synthetic: bool,
    data_dir: str | None,
    seed: int,
) -> None:
    """Evaluate a trained model checkpoint."""
    from training.export import load_for_inference  # noqa: PLC0415

    click.echo(f"Loading checkpoint: {checkpoint}")
    model = load_for_inference(checkpoint, game=game)  # type: ignore[arg-type]
    model.eval()

    if game == "chess":
        from training.data.chessbench_loader import make_chess_loader  # noqa: PLC0415
        num_batches = max(1, num_positions // batch_size)
        loader = make_chess_loader(
            synthetic=synthetic,
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
            max_batches=num_batches,
        )
    else:
        from training.data.dlshogi_loader import make_shogi_loader  # noqa: PLC0415
        num_batches = max(1, num_positions // batch_size)
        loader = make_shogi_loader(
            synthetic=synthetic,
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
            max_batches=num_batches,
        )

    policy_correct = 0
    total_positions = 0
    value_mse_sum = 0.0

    for batch in loader:
        outputs = model(batch.positions, game)  # type: ignore[arg-type]

        # Policy top-1 accuracy
        pred_idx = mx.argmax(outputs["policy"], axis=-1)  # [B]
        true_idx = mx.argmax(batch.policy_target, axis=-1)  # [B]
        mx.eval(pred_idx, true_idx)
        correct = int((np.array(pred_idx) == np.array(true_idx)).sum())
        policy_correct += correct

        # Value MSE
        value_pred = np.array(outputs["value"].astype(mx.float32))
        value_true = np.array(batch.value_target)
        value_mse_sum += float(((value_pred - value_true) ** 2).mean())

        total_positions += batch.positions.shape[0]

    top1_acc = policy_correct / max(1, total_positions)
    avg_value_mse = value_mse_sum / max(1, (total_positions // batch_size))

    click.echo(f"\nResults over {total_positions} positions:")
    click.echo(f"  Policy top-1 accuracy: {top1_acc:.4f} ({policy_correct}/{total_positions})")
    click.echo(f"  Value MSE:             {avg_value_mse:.4f}")


if __name__ == "__main__":
    main()
