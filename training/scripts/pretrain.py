"""
Pre-training entry-point for ChessShogiTransformer.

Usage::

    # Single-game (unchanged):
    uv run python -m training.scripts.pretrain \\
        --game chess \\
        --steps 10000 \\
        --batch-size 128 \\
        --checkpoint-dir checkpoints/run1 \\
        --synthetic

    # Joint distillation (Phase 3 XGD):
    uv run python -m training.scripts.pretrain \\
        --joint-distill \\
        --chess-data /data/chessbench \\
        --shogi-data /data/hcpe \\
        --steps 100000 \\
        --batch-size 128 \\
        --checkpoint-dir checkpoints/joint_run1

For a quick smoke-test::

    uv run python -m training.scripts.pretrain --game chess --steps 3 --batch-size 8 --synthetic
    uv run python -m training.scripts.pretrain --joint-distill --synthetic --steps 6 --batch-size 4
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
@click.option("--game", type=click.Choice(["chess", "shogi"]), default="chess", show_default=True,
              help="Game for single-game training. Ignored when --joint-distill is set.")
@click.option("--steps", type=int, default=10_000, show_default=True, help="Number of gradient steps.")
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True, help="Peak learning rate.")
@click.option("--weight-decay", type=float, default=0.1, show_default=True)
@click.option("--warmup-steps", type=int, default=500, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--checkpoint-dir", type=str, default="checkpoints", show_default=True)
@click.option("--checkpoint-every", type=int, default=1000, show_default=True)
@click.option("--synthetic", is_flag=True, default=False, help="Use synthetic data (no download needed).")
@click.option("--data-dir", type=str, default=None, help="Data directory for single-game real data.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--log-every", type=int, default=1, show_default=True)
@click.option(
    "--n-layers", type=int, default=12, show_default=True,
    help="Number of transformer layers."
)
@click.option("--d-model", type=int, default=512, show_default=True)
@click.option("--n-heads", type=int, default=8, show_default=True)
@click.option("--ffn-dim", type=int, default=2048, show_default=True)
@click.option("--schedule", type=click.Choice(["cosine", "wsd"]), default="cosine", show_default=True,
              help="LR schedule: cosine decay or warmup-stable-decay (linear-flat-linear).")
@click.option("--wsd-decay-frac", type=float, default=0.20, show_default=True,
              help="Fraction of training spent in the linear decay tail (WSD only).")
@click.option("--w-policy", type=float, default=1.0, show_default=True)
@click.option("--w-value", type=float, default=1.0, show_default=True)
@click.option("--w-moves-left", type=float, default=0.1, show_default=True)
# ---------------------------------------------------------------------------
# Joint distillation flags (Phase 3 XGD)
# ---------------------------------------------------------------------------
@click.option("--joint-distill", is_flag=True, default=False,
              help="Enable joint chess+shogi distillation (XGD Phase 3).  "
                   "When set, --game is overridden to 'both' and both "
                   "--chess-data and --shogi-data (or --synthetic) are required.")
@click.option("--chess-data", type=str, default=None,
              help="Path to chess data directory (required for joint distillation "
                   "unless --synthetic).")
@click.option("--shogi-data", type=str, default=None,
              help="Path to shogi .hcpe file or directory (required for joint "
                   "distillation unless --synthetic).")
@click.option("--interleave", type=click.Choice(["round_robin", "weighted"]),
              default="round_robin", show_default=True,
              help="Batch interleave strategy for joint distillation.")
@click.option("--shogi-weight", type=float, default=0.5, show_default=True,
              help="Probability of drawing a shogi batch when --interleave weighted.")
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
    schedule: str,
    wsd_decay_frac: float,
    w_policy: float,
    w_value: float,
    w_moves_left: float,
    joint_distill: bool,
    chess_data: str | None,
    shogi_data: str | None,
    interleave: str,
    shogi_weight: float,
) -> None:
    """Pre-train ChessShogiTransformer on chess, shogi, or both (joint distillation)."""
    # Late imports to keep startup fast
    from training.models.transformer import ChessShogiTransformer  # noqa: PLC0415

    # ------------------------------------------------------------------
    # Joint distillation path
    # ------------------------------------------------------------------
    if joint_distill:
        _run_joint(
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            grad_clip=grad_clip,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            synthetic=synthetic,
            chess_data=chess_data,
            shogi_data=shogi_data,
            seed=seed,
            log_every=log_every,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            schedule=schedule,
            wsd_decay_frac=wsd_decay_frac,
            w_policy=w_policy,
            w_value=w_value,
            w_moves_left=w_moves_left,
            interleave=interleave,
            shogi_weight=shogi_weight,
        )
        return

    # ------------------------------------------------------------------
    # Single-game path (backwards-compatible, unchanged logic)
    # ------------------------------------------------------------------
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
        w_policy=w_policy,
        w_value=w_value,
        w_moves_left=w_moves_left,
        schedule=schedule,  # type: ignore[arg-type]
        wsd_decay_frac=wsd_decay_frac,
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


# ---------------------------------------------------------------------------
# Joint distillation helper (keeps main() readable)
# ---------------------------------------------------------------------------

def _run_joint(
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    grad_clip: float,
    checkpoint_dir: str,
    checkpoint_every: int,
    synthetic: bool,
    chess_data: str | None,
    shogi_data: str | None,
    seed: int,
    log_every: int,
    n_layers: int,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
    schedule: str,
    wsd_decay_frac: float,
    w_policy: float,
    w_value: float,
    w_moves_left: float,
    interleave: str,
    shogi_weight: float,
) -> None:
    """Execute the joint chess+shogi distillation training run."""
    from training.data.chessbench_loader import make_chess_loader  # noqa: PLC0415
    from training.data.dlshogi_loader import make_shogi_loader  # noqa: PLC0415
    from training.data.joint_loader import JointLoader  # noqa: PLC0415
    from training.models.transformer import ChessShogiTransformer  # noqa: PLC0415
    from training.trainers.supervised import JointSupervisedTrainer  # noqa: PLC0415

    if not synthetic and (chess_data is None or shogi_data is None):
        click.echo(
            "ERROR: --chess-data and --shogi-data are required for joint distillation "
            "unless --synthetic is set.",
            err=True,
        )
        sys.exit(1)

    click.echo("Pre-training joint chess+shogi transformer (XGD Phase 3)")
    click.echo(f"  layers={n_layers}, d_model={d_model}, heads={n_heads}, ffn={ffn_dim}")
    click.echo(f"  steps={steps}, batch={batch_size}, lr={lr}, warmup={warmup_steps}")
    click.echo(f"  interleave={interleave}, shogi_weight={shogi_weight:.2f}")
    click.echo(f"  synthetic={synthetic}, seed={seed}")

    # Build joint model
    model = ChessShogiTransformer(
        game="both",
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
    )
    param_count = model.count_parameters()
    click.echo(f"  parameters={param_count:,}")

    # Each game loader yields 2× the total steps to account for alternation
    per_game_batches = steps  # both iterators contribute ~steps/2 batches each

    chess_loader = make_chess_loader(
        synthetic=synthetic,
        data_dir=chess_data,
        batch_size=batch_size,
        seed=seed,
        max_batches=per_game_batches,
    )
    shogi_loader = make_shogi_loader(
        synthetic=synthetic,
        data_dir=shogi_data,
        batch_size=batch_size,
        seed=seed + 1,  # different seed for diversity
        max_batches=per_game_batches,
    )

    joint_loader = JointLoader(
        chess_loader=iter(chess_loader),
        shogi_loader=iter(shogi_loader),
        interleave=interleave,  # type: ignore[arg-type]
        weights=(1.0 - shogi_weight, shogi_weight),
        seed=seed,
    )

    trainer = JointSupervisedTrainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        warmup_steps=warmup_steps,
        total_steps=steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        w_policy=w_policy,
        w_value=w_value,
        w_moves_left=w_moves_left,
        schedule=schedule,  # type: ignore[arg-type]
        wsd_decay_frac=wsd_decay_frac,
    )

    metrics = trainer.train(
        data_iter=joint_loader,
        num_steps=steps,
        log_every=log_every,
        verbose=True,
    )

    final_path = trainer.save_final(
        str(Path(checkpoint_dir) / "joint_final.safetensors")
    )
    click.echo(f"Saved final joint weights to {final_path}")

    if metrics:
        losses = [m["loss"] for m in metrics]
        click.echo(f"Final loss: {losses[-1]:.4f}  (initial: {losses[0]:.4f})")

        # Report per-game breakdown from history
        chess_hist = [m for m in trainer.history if m.get("game", -1) == 0.0]
        shogi_hist = [m for m in trainer.history if m.get("game", -1) == 1.0]
        if chess_hist:
            avg_chess = sum(m["loss"] for m in chess_hist[-50:]) / len(chess_hist[-50:])
            click.echo(f"  Chess avg loss (last 50): {avg_chess:.4f}")
        if shogi_hist:
            avg_shogi = sum(m["loss"] for m in shogi_hist[-50:]) / len(shogi_hist[-50:])
            click.echo(f"  Shogi avg loss (last 50): {avg_shogi:.4f}")


if __name__ == "__main__":
    main()
