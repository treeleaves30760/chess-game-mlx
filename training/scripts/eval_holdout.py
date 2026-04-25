"""
Evaluate a trained shogi checkpoint on a held-out hcpe set.

Reports:
    * Policy top-1 accuracy (best move match)
    * Policy top-5 accuracy
    * Policy NLL (cross-entropy in nats)
    * Value MSE vs game-outcome (signed from side-to-move POV)

Usage::

    uv run python training/scripts/eval_holdout.py \\
        --ckpt checkpoints/shogi_40m/final.safetensors \\
        --holdout data/dlshogi/floodgate_2025_R2700_holdout.hcpe \\
        --max-positions 50000
"""

from __future__ import annotations

import math
import time

import click
import numpy as np


@click.command()
@click.option("--ckpt", type=str, required=True)
@click.option("--holdout", type=str, required=True, help="Path to .hcpe file.")
@click.option("--max-positions", type=int, default=50_000, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--n-layers", type=int, default=12, show_default=True)
@click.option("--d-model", type=int, default=512, show_default=True)
@click.option("--n-heads", type=int, default=8, show_default=True)
@click.option("--ffn-dim", type=int, default=2048, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    ckpt: str,
    holdout: str,
    max_positions: int,
    batch_size: int,
    n_layers: int,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
    seed: int,
) -> None:
    import mlx.core as mx  # noqa: PLC0415
    from training.models.transformer import ChessShogiTransformer  # noqa: PLC0415
    from training.data.dlshogi_loader import (  # noqa: PLC0415
        DlshogiLoader,
        SHOGI_NUM_MOVES,
    )

    click.echo(f"Loading model from {ckpt}")
    model = ChessShogiTransformer(
        game="shogi",
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
    )
    model.load_weights(ckpt)
    model.eval()

    n_batches = max_positions // batch_size
    click.echo(f"Evaluating {n_batches} batches × {batch_size} = {n_batches*batch_size:,} positions")

    loader = DlshogiLoader(
        path=holdout,
        batch_size=batch_size,
        seed=seed,
        value_lambda=0.0,  # pure outcome value for evaluation
        max_batches=n_batches,
        skip_invalid=True,
    )

    top1_hits = 0
    top5_hits = 0
    total = 0
    nll_sum = 0.0
    value_sq_err_sum = 0.0
    value_count = 0

    t0 = time.perf_counter()
    for batch_i, batch in enumerate(loader):
        out = model(batch.positions, game="shogi")
        logits = out["policy"].astype(mx.float32)
        values = out["value"].astype(mx.float32)

        # Compute log-softmax
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        log_probs_np = np.asarray(log_probs)  # [B, 2187]
        target_np = np.asarray(batch.policy_target)  # [B, 2187] one-hot

        # Top-k via argmax / argpartition
        target_idx = target_np.argmax(axis=-1)
        argmax = log_probs_np.argmax(axis=-1)
        top5 = np.argpartition(-log_probs_np, 5, axis=-1)[:, :5]

        target_present = target_np.sum(axis=-1) > 0  # mask invalid records
        for b in range(batch_size):
            if not target_present[b]:
                continue
            tgt = int(target_idx[b])
            total += 1
            if argmax[b] == tgt:
                top1_hits += 1
            if tgt in top5[b]:
                top5_hits += 1
            nll_sum -= float(log_probs_np[b, tgt])

        # Value MSE
        v_pred = np.asarray(values).flatten()
        v_tgt = np.asarray(batch.value_target).flatten()
        value_sq_err_sum += float(((v_pred - v_tgt) ** 2).sum())
        value_count += v_pred.shape[0]

        if (batch_i + 1) % 50 == 0:
            click.echo(f"  batch {batch_i+1}/{n_batches}, elapsed {time.perf_counter()-t0:.1f}s")

    elapsed = time.perf_counter() - t0
    click.echo("=" * 60)
    click.echo(f"Evaluated {total:,} positions in {elapsed:.1f}s ({total/elapsed:.0f}/s)")
    if total > 0:
        click.echo(f"  policy top-1 acc : {top1_hits/total*100:.2f}%  (random ≈ {100/SHOGI_NUM_MOVES:.3f}%)")
        click.echo(f"  policy top-5 acc : {top5_hits/total*100:.2f}%")
        nll = nll_sum / total
        click.echo(f"  policy NLL       : {nll:.3f} nats  (uniform-2187 = {math.log(SHOGI_NUM_MOVES):.3f})")
        click.echo(f"  policy perplexity: {math.exp(nll):.1f}")
    if value_count > 0:
        click.echo(f"  value MSE        : {value_sq_err_sum/value_count:.4f}")
        click.echo(f"  value RMSE       : {math.sqrt(value_sq_err_sum/value_count):.4f}")


if __name__ == "__main__":
    main()
