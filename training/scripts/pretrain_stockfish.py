"""Real training run using Stockfish 18 as teacher.

Pipeline:
  1. Pre-label N random positions with Stockfish (depth 10) → .npz dataset
  2. Train the 40M ChessShogiTransformer on those labels with cosine LR
  3. Export weights to safetensors for the C++ engine to load

Usage:
  uv run python -m training.scripts.pretrain_stockfish \
      --positions 2000 --epochs 50 --batch-size 64 \
      --out-dir checkpoints/stockfish_2k
"""

from __future__ import annotations

from pathlib import Path

import click
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from training.data.stockfish_teacher import (
    StockfishTeacher,
    load_prelabeled_dataset,
    save_prelabeled_dataset,
)
from training.export import export_model
from training.losses import combined_loss
from training.models.transformer import ChessShogiTransformer


@click.command()
@click.option("--positions", type=int, default=2000, help="Number of stockfish-annotated positions")
@click.option("--epochs", type=int, default=30)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1e-4)
@click.option("--warmup-steps", type=int, default=100)
@click.option("--depth", type=int, default=10, help="Stockfish search depth for annotation")
@click.option("--data-path", type=click.Path(path_type=Path), default=Path("data/stockfish_labels.npz"))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("checkpoints/stockfish"))
@click.option("--regenerate", is_flag=True, help="Re-annotate dataset even if it exists")
@click.option("--d-model", type=int, default=512)
@click.option("--layers", type=int, default=12)
@click.option("--seed", type=int, default=42)
def main(
    positions: int,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    depth: int,
    data_path: Path,
    out_dir: Path,
    regenerate: bool,
    d_model: int,
    layers: int,
    seed: int,
) -> None:
    """Train on Stockfish-annotated random positions."""
    mx.random.seed(seed)

    # ---- step 1: ensure we have labelled data ----
    if regenerate or not data_path.exists():
        print(f"Annotating {positions} positions with Stockfish (depth={depth})...")
        with StockfishTeacher(depth=depth, threads=2) as teacher:
            save_prelabeled_dataset(data_path, teacher, positions, seed=seed)
    else:
        print(f"Re-using existing labels at {data_path}")

    import numpy as np
    data = np.load(data_path)
    n_positions = len(data["x"])
    steps_per_epoch = max(1, n_positions // batch_size)
    total_steps = steps_per_epoch * epochs
    print(f"  dataset: {n_positions} positions")
    print(f"  epochs: {epochs}, steps/epoch: {steps_per_epoch}, total steps: {total_steps}")

    # ---- step 2: build model ----
    model = ChessShogiTransformer(
        n_layers=layers,
        d_model=d_model,
        n_heads=max(1, d_model // 64),
        ffn_dim=d_model * 4,
    )
    mx.eval(model.parameters())
    total_params = model.count_parameters() if hasattr(model, "count_parameters") else 0
    print(f"  model params: {total_params:,}")

    # ---- step 3: optimizer + LR schedule ----
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    num_moves = 1858

    def loss_fn(m, x, p_idx, v_target, ml_target):
        out = m(x, game="chess")
        # p_idx is [B] sparse indices; one-hot into [B, num_moves]
        p_target = mx.zeros((p_idx.shape[0], num_moves))
        p_target[mx.arange(p_idx.shape[0]), p_idx] = 1.0
        # Ensure value and moves_left targets are [B, 1] to match head output
        v_t = v_target.reshape(-1, 1)
        ml_t = ml_target.reshape(-1, 1)
        total, losses = combined_loss(
            policy_logits=out["policy"],
            policy_target=p_target,
            value_pred=out["value"],
            value_target=v_t,
            moves_left_pred=out["moves_left"],
            moves_left_target=ml_t,
        )
        return total, losses

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # ---- step 4: training loop ----
    # Uses epoch-based iteration: one full pass per epoch with per-epoch seeds
    # so the shuffle is different each epoch (important for generalisation).
    # mx.eval() deferred to log intervals to avoid per-step GPU sync stalls.
    LOG_EVERY = 50
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training for {total_steps} steps (target loss convergence)...")
    step = 0
    best_loss = float("inf")
    loss = None
    losses: dict = {}

    # Load dataset once into memory for fast per-epoch access
    _data = np.load(data_path)
    _x_all = _data["x"]  # [N, 64, 19]
    _p_all = _data["p"]  # [N]
    _v_all = _data["v"]  # [N]
    _m_all = _data["m"]  # [N]
    _N = len(_x_all)

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        idx = rng.permutation(_N)
        epoch_done = False

        for s in range(0, _N - batch_size + 1, batch_size):
            b = idx[s : s + batch_size]
            x = mx.array(_x_all[b])
            p = mx.array(_p_all[b])
            v = mx.array(_v_all[b])
            m = mx.array(_m_all[b])

            optimizer.learning_rate = lr_schedule(step)
            (loss, losses), grads = loss_and_grad_fn(model, x, p, v, m)
            grads, _gnorm = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)

            step += 1
            if step % LOG_EVERY == 0 or step == 1:
                mx.eval(model.parameters(), optimizer.state)
                print(
                    f"  step {step:5d}/{total_steps} | epoch {epoch+1}/{epochs} | "
                    f"total {float(loss):7.4f} | policy {float(losses['policy']):7.4f} | "
                    f"value {float(losses['value']):6.4f} | lr {optimizer.learning_rate:.2e}"
                )
            if step >= total_steps:
                epoch_done = True
                break

        # Per-epoch checkpoint — force sync
        if loss is not None:
            mx.eval(model.parameters(), optimizer.state)
            cur_loss = float(loss)
            if cur_loss < best_loss:
                best_loss = cur_loss
                ckpt_path = out_dir / "best.safetensors"
                export_model(model, ckpt_path, game="chess")
                print(f"  [checkpoint] epoch {epoch+1}/{epochs} best ({cur_loss:.4f}) -> {ckpt_path}")

        if epoch_done:
            break

    # Final export
    final_path = out_dir / "final.safetensors"
    export_model(model, final_path, game="chess")
    print(f"\nDone. Final model: {final_path}")
    if loss is not None:
        print(f"Final loss: {float(loss):.4f}")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
