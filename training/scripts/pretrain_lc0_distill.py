"""Distillation training from LC0-labelled positions.

Thin wrapper around `pretrain_stockfish.py`'s code that exists solely to
provide clearer defaults for distillation runs (lower moves_left loss weight,
slightly higher learning rate, and a clearly-named output directory).

Usage:
    uv run python training/scripts/pretrain_lc0_distill.py \
        --data-path data/lc0_labels_20k.npz \
        --epochs 10 --batch-size 128 --lr 2e-4 \
        --d-model 512 --layers 12 \
        --out-dir checkpoints/chess_40m_lc0distill_20k
"""

from __future__ import annotations

import math
from pathlib import Path

import click
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from training.data.stockfish_teacher import load_prelabeled_dataset
from training.export import export_model
from training.losses import combined_loss
from training.models.transformer import ChessShogiTransformer


@click.command()
@click.option(
    "--data-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Pre-labelled dataset .npz (from label_lc0.py).",
)
@click.option("--epochs", type=int, default=10)
@click.option("--batch-size", type=int, default=128)
@click.option("--lr", type=float, default=2e-4)
@click.option("--warmup-steps", type=int, default=150)
@click.option("--d-model", type=int, default=512)
@click.option("--layers", type=int, default=12)
@click.option("--ffn-dim", type=int, default=0, help="0 = auto (4*d_model)")
@click.option("--heads", type=int, default=0, help="0 = auto (d_model // 64)")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("checkpoints/chess_40m_lc0distill"),
)
@click.option(
    "--w-moves-left",
    type=float,
    default=0.001,
    help="Weight on moves-left loss. Low default because distillation value "
    "labels span [-1, +1] and moves-left labels span [0, 200] — the scale "
    "mismatch makes the default 0.1 dominate the loss budget.",
)
@click.option("--seed", type=int, default=42)
@click.option("--log-every", type=int, default=25)
def main(
    data_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    d_model: int,
    layers: int,
    ffn_dim: int,
    heads: int,
    out_dir: Path,
    w_moves_left: float,
    seed: int,
    log_every: int,
) -> None:
    """Train ``ChessShogiTransformer`` on pre-labelled LC0 data."""
    mx.random.seed(seed)

    if ffn_dim == 0:
        ffn_dim = 4 * d_model
    if heads == 0:
        heads = max(1, d_model // 64)

    # ---- load dataset ----
    data = np.load(data_path)
    n_positions = len(data["x"])
    steps_per_epoch = max(1, n_positions // batch_size)
    total_steps = steps_per_epoch * epochs
    print(f"dataset: {data_path} ({n_positions} positions)")
    print(f"  epochs: {epochs}, batch: {batch_size}, steps/epoch: {steps_per_epoch}")
    print(f"  total steps: {total_steps}")
    print(f"  value range: [{data['v'].min():+.3f}, {data['v'].max():+.3f}]")
    print(f"  moves-left mean: {data['m'].mean():.1f}  (loss weight={w_moves_left})")

    # ---- build model ----
    model = ChessShogiTransformer(
        game="chess",
        n_layers=layers,
        d_model=d_model,
        n_heads=heads,
        ffn_dim=ffn_dim,
    )
    mx.eval(model.parameters())
    total_params = model.count_parameters()
    print(f"model: {layers}L × {d_model}d × {heads}h, ffn={ffn_dim}")
    print(f"  params: {total_params:,}")

    # ---- optimizer + schedule ----
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    num_moves = 1858

    def loss_fn(m, x, p_idx, v_target, ml_target):
        out = m(x, game="chess")
        p_target = mx.zeros((p_idx.shape[0], num_moves))
        p_target[mx.arange(p_idx.shape[0]), p_idx] = 1.0
        v_t = v_target.reshape(-1, 1)
        ml_t = ml_target.reshape(-1, 1)
        total, losses = combined_loss(
            policy_logits=out["policy"],
            policy_target=p_target,
            value_pred=out["value"],
            value_target=v_t,
            moves_left_pred=out["moves_left"],
            moves_left_target=ml_t,
            w_moves_left=w_moves_left,
        )
        return total, losses

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # ---- training loop ----
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best_loss = float("inf")
    loss = None
    for epoch in range(epochs):
        for x, p, v, m in load_prelabeled_dataset(
            data_path, batch_size, shuffle=True, seed=seed + epoch
        ):
            optimizer.learning_rate = lr_schedule(step)
            (loss, losses), grads = loss_and_grad_fn(model, x, p, v, m)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)

            step += 1
            if step % log_every == 0 or step == 1:
                mx.eval(model.parameters(), optimizer.state)
                print(
                    f"  step {step:5d}/{total_steps} | epoch {epoch+1}/{epochs} | "
                    f"total {float(loss):7.4f} | policy {float(losses['policy']):7.4f} | "
                    f"value {float(losses['value']):6.4f} | "
                    f"ml {float(losses['moves_left']):7.2f} | "
                    f"lr {optimizer.learning_rate:.2e}",
                    flush=True,
                )
            # Periodic checkpoint: write current weights every
            # CKPT_EVERY steps so a kill mid-training still leaves
            # usable weights.
            CKPT_EVERY = 100
            if step % CKPT_EVERY == 0:
                mx.eval(model.parameters(), optimizer.state)
                ckpt_path = out_dir / "latest.safetensors"
                export_model(model, ckpt_path, game="chess")

            if step >= total_steps:
                break

        mx.eval(model.parameters(), optimizer.state)
        if loss is not None:
            cur_loss = float(loss)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_path = out_dir / "best.safetensors"
                export_model(model, best_path, game="chess")
                print(f"  [checkpoint] epoch {epoch+1} best ({cur_loss:.4f}) -> {best_path}")
        if step >= total_steps:
            break

    final_path = out_dir / "final.safetensors"
    export_model(model, final_path, game="chess")
    print(f"\nDone. Final model: {final_path}")
    if loss is not None:
        print(f"Final loss: {float(loss):.4f}")


if __name__ == "__main__":
    main()
