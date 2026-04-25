"""
Pre-training entry-point for ChessShogiTransformer.

Usage::

    uv run python -m training.scripts.pretrain \\
        --game chess \\
        --steps 10000 \\
        --batch-size 128 \\
        --checkpoint-dir checkpoints/run1 \\
        --synthetic

For a quick smoke-test::

    uv run python -m training.scripts.pretrain --game chess --steps 3 --batch-size 8 --synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import mlx.core as mx

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("pretrain")
@click.option("--game", type=click.Choice(["chess", "shogi"]), default="chess", show_default=True)
@click.option("--steps", type=int, default=10_000, show_default=True, help="Number of gradient steps.")
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True, help="Peak learning rate.")
@click.option("--weight-decay", type=float, default=0.1, show_default=True)
@click.option("--warmup-steps", type=int, default=500, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--checkpoint-dir", type=str, default="checkpoints", show_default=True)
@click.option("--checkpoint-every", type=int, default=1000, show_default=True)
@click.option("--synthetic", is_flag=True, default=False, help="Use synthetic data (no download needed).")
@click.option("--data-dir", type=str, default=None, help="Data directory for real data.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--log-every", type=int, default=1, show_default=True)
@click.option(
    "--n-layers", type=int, default=12, show_default=True,
    help="Number of transformer layers."
)
@click.option("--d-model", type=int, default=512, show_default=True)
@click.option("--n-heads", type=int, default=8, show_default=True)
@click.option("--ffn-dim", type=int, default=2048, show_default=True)
def main(
    game: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    grad_clip: float,
    checkpoint_dir: str,
    checkpoint_every: int,
    synthetic: bool,
    data_dir: str | None,
    seed: int,
    log_every: int,
    n_layers: int,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
) -> None:
    """Pre-train ChessShogiTransformer on chess or shogi data."""
    # Late imports to keep startup fast
    from training.models.transformer import ChessShogiTransformer  # noqa: PLC0415
    from training.trainers.supervised import SupervisedTrainer  # noqa: PLC0415

    if not synthetic and data_dir is None:
        click.echo("ERROR: --data-dir required unless --synthetic is set.", err=True)
        sys.exit(1)

    click.echo(f"Pre-training {game.upper()} transformer")
    click.echo(f"  layers={n_layers}, d_model={d_model}, heads={n_heads}, ffn={ffn_dim}")
    click.echo(f"  steps={steps}, batch={batch_size}, lr={lr}, warmup={warmup_steps}")
    click.echo(f"  synthetic={synthetic}, seed={seed}")

    # Build model
    model = ChessShogiTransformer(
        game=game,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
    )
    param_count = model.count_parameters()
    click.echo(f"  parameters={param_count:,}")

    # Data loader
    if game == "chess":
        from training.data.chessbench_loader import make_chess_loader  # noqa: PLC0415
        loader = make_chess_loader(
            synthetic=synthetic,
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
            max_batches=steps,
        )
    else:
        from training.data.dlshogi_loader import make_shogi_loader  # noqa: PLC0415
        loader = make_shogi_loader(
            synthetic=synthetic,
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
            max_batches=steps,
        )

    # Trainer
    trainer = SupervisedTrainer(
        model=model,
        game=game,  # type: ignore[arg-type]
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        warmup_steps=warmup_steps,
        total_steps=steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
    )

    # Run
    metrics = trainer.train(
        data_iter=loader,
        num_steps=steps,
        log_every=log_every,
        verbose=True,
    )

    # Save final
    final_path = trainer.save_final(
        str(Path(checkpoint_dir) / f"{game}_final.safetensors")
    )
    click.echo(f"Saved final weights to {final_path}")

    # Report final loss
    if metrics:
        losses = [m["loss"] for m in metrics]
        click.echo(f"Final loss: {losses[-1]:.4f}  (initial: {losses[0]:.4f})")


if __name__ == "__main__":
    main()
