"""Soft-label distillation training from LC0-annotated positions.

Uses the ``p_top_idx`` / ``p_top_prob`` fields saved by the upgraded
``lc0_teacher.py`` (top-K soft policy per position). Loss is KL-divergence
between the model's policy logits and LC0's soft policy — dramatically more
informative than one-hot-on-argmax, because the entire shape of LC0's prior
is used.

Expected Elo lift over hard-label distillation: +400-800 Elo at the same
dataset size, because we transmit ≈ log2(K) bits per position instead of 1.

Usage:
    uv run python training/scripts/pretrain_lc0_soft.py \
        --data-path data/lc0_labels_100k.npz \
        --epochs 15 --batch-size 128 --lr 2e-4 \
        --d-model 512 --layers 12 \
        --out-dir checkpoints/chess_40m_lc0soft_100k
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import click
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from training.export import export_model
from training.models.transformer import ChessShogiTransformer


NUM_MOVES = 1858


def _load_soft_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the soft-label dataset."""
    data = np.load(path)
    required = {"x", "v", "m", "p_top_idx", "p_top_prob"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(
            f"{path} missing fields {missing}; is this a soft-label dataset "
            "from the upgraded lc0_teacher? Re-label with "
            "training/scripts/label_lc0.py after pulling the latest code."
        )
    return {k: data[k] for k in data.files}


def _iter_soft_batches(
    data: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
):
    """Infinite iterator of (x, p_top_idx, p_top_prob, v, m) MLX batches."""
    n = len(data["x"])
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n - batch_size + 1, batch_size):
            b = idx[s : s + batch_size]
            yield (
                mx.array(data["x"][b]),
                mx.array(data["p_top_idx"][b]),
                mx.array(data["p_top_prob"][b]),
                mx.array(data["v"][b]),
                mx.array(data["m"][b]),
            )


def _build_soft_target(
    p_top_idx: mx.array,  # [B, K] int32 (-1 pad)
    p_top_prob: mx.array,  # [B, K] float32
) -> mx.array:
    """Scatter padded top-K soft policy into a full [B, NUM_MOVES] tensor.

    Pad positions (idx == -1) contribute 0 probability. The result is a valid
    probability distribution per row (sums to ~1 across non-pad entries).
    """
    B, K = p_top_idx.shape
    # Mask out the pad entries (-1) so the scatter below is well-defined.
    mask = p_top_idx >= 0  # [B, K] bool
    # Replace -1 with 0 so we can scatter; masked probabilities are 0 anyway.
    safe_idx = mx.where(mask, p_top_idx, mx.zeros_like(p_top_idx))
    safe_prob = mx.where(mask, p_top_prob, mx.zeros_like(p_top_prob))

    # Use one-hot + weighted sum to scatter (MLX lacks scatter-add on axis).
    # Shape of one_hot: [B, K, NUM_MOVES] bool/float.
    one_hot = mx.eye(NUM_MOVES)[safe_idx]  # [B, K, NUM_MOVES]
    # Multiply each column's probability
    weighted = one_hot * safe_prob[:, :, None]  # [B, K, NUM_MOVES]
    # Sum across K to get the soft target
    soft_target = weighted.sum(axis=1)  # [B, NUM_MOVES]
    return soft_target


def _kl_loss(logits: mx.array, target: mx.array, eps: float = 1e-9) -> mx.array:
    """KL(target || softmax(logits)), averaged across the batch.

    We minimize cross-entropy CE(target, p_model) which equals
    KL + H(target); since H(target) is constant w.r.t. parameters,
    minimizing CE minimizes KL.
    """
    # log_softmax: logits - logsumexp
    logsumexp = mx.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = logits - logsumexp
    # CE = -sum(target * log_probs)
    ce_per_row = -(target * log_probs).sum(axis=-1)
    return ce_per_row.mean()


@click.command()
@click.option("--data-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--epochs", type=int, default=15)
@click.option("--batch-size", type=int, default=128)
@click.option("--lr", type=float, default=2e-4)
@click.option("--warmup-steps", type=int, default=300)
@click.option("--d-model", type=int, default=512)
@click.option("--layers", type=int, default=12)
@click.option("--ffn-dim", type=int, default=0, help="0 = auto (4*d_model)")
@click.option("--heads", type=int, default=0, help="0 = auto (d_model // 64)")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("checkpoints/chess_40m_lc0soft"),
)
@click.option(
    "--w-value",
    type=float,
    default=1.0,
    help="Weight on value-MSE loss. LC0 values are well-calibrated → keep at 1.",
)
@click.option(
    "--w-moves-left",
    type=float,
    default=1e-4,
    help="Weight on moves-left loss. Very low because moves_left labels span "
    "[0, 200] and dominate the scale otherwise.",
)
@click.option("--seed", type=int, default=42)
@click.option("--log-every", type=int, default=25)
@click.option("--ckpt-every", type=int, default=200, help="Save latest weights every N steps")
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
    w_value: float,
    w_moves_left: float,
    seed: int,
    log_every: int,
    ckpt_every: int,
) -> None:
    """Soft-label KL distillation training."""
    mx.random.seed(seed)

    if ffn_dim == 0:
        ffn_dim = 4 * d_model
    if heads == 0:
        heads = max(1, d_model // 64)

    # ---- load dataset ----
    print(f"Loading soft-label dataset from {data_path}...")
    data = _load_soft_dataset(data_path)
    n = len(data["x"])
    steps_per_epoch = max(1, n // batch_size)
    total_steps = steps_per_epoch * epochs
    # Sanity stats
    avg_top1 = data["p_top_prob"][:, 0].mean()
    nonzero_entries = (data["p_top_prob"] > 0).sum(axis=1).mean()
    print(f"  positions:     {n}")
    print(f"  epochs:        {epochs}  (batch={batch_size}, steps/epoch={steps_per_epoch})")
    print(f"  total steps:   {total_steps}")
    print(f"  avg top-1 p:   {avg_top1:.3f}  (should be ~0.3-0.5 for trained LC0)")
    print(f"  avg top-K len: {nonzero_entries:.1f}  (=# legal moves with non-zero prob)")
    print(f"  value range:   [{data['v'].min():+.3f}, {data['v'].max():+.3f}]")

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
    print(f"model: {layers}L × {d_model}d × {heads}h × ffn={ffn_dim}")
    print(f"  params: {total_params:,}")

    # ---- optimizer + schedule ----
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, x, p_top_idx, p_top_prob, v_target, ml_target):
        out = m(x, game="chess")
        # Build soft target and compute KL / CE
        soft_target = _build_soft_target(p_top_idx, p_top_prob)
        p_loss = _kl_loss(out["policy"], soft_target)
        # Value MSE (targets are white-POV from LC0)
        v_pred = out["value"].reshape(-1)
        v_loss = ((v_pred - v_target) ** 2).mean()
        # Moves-left L2
        ml_pred = out["moves_left"].reshape(-1)
        ml_loss = ((ml_pred - ml_target) ** 2).mean()
        total = p_loss + w_value * v_loss + w_moves_left * ml_loss
        return total, {"policy": p_loss, "value": v_loss, "moves_left": ml_loss}

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # ---- training loop ----
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best_loss = float("inf")
    last_loss = None
    t0 = time.time()

    for epoch in range(epochs):
        for x, pti, ptp, v, ml in _iter_soft_batches(
            data, batch_size, shuffle=True, seed=seed + epoch
        ):
            optimizer.learning_rate = lr_schedule(step)
            (total, parts), grads = loss_and_grad_fn(model, x, pti, ptp, v, ml)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)

            step += 1
            last_loss = float(total)
            if step % log_every == 0 or step == 1:
                mx.eval(model.parameters(), optimizer.state)
                elapsed = time.time() - t0
                rate = step / max(elapsed, 1e-6)
                eta_min = (total_steps - step) / max(rate, 1e-6) / 60
                print(
                    f"  step {step:5d}/{total_steps} | "
                    f"loss {last_loss:6.4f} | p {float(parts['policy']):6.4f} | "
                    f"v {float(parts['value']):5.3f} | ml {float(parts['moves_left']):7.1f} | "
                    f"lr {optimizer.learning_rate:.2e} | "
                    f"{rate:.2f} st/s | eta {eta_min:.1f} min",
                    flush=True,
                )

            if step % ckpt_every == 0:
                mx.eval(model.parameters(), optimizer.state)
                export_model(model, out_dir / "latest.safetensors", game="chess")

            if step >= total_steps:
                break
        # End-of-epoch checkpoint
        mx.eval(model.parameters(), optimizer.state)
        if last_loss is not None and last_loss < best_loss:
            best_loss = last_loss
            export_model(model, out_dir / "best.safetensors", game="chess")
            print(f"  [checkpoint] epoch {epoch+1} best ({best_loss:.4f}) -> best.safetensors", flush=True)
        if step >= total_steps:
            break

    # Final export
    mx.eval(model.parameters(), optimizer.state)
    export_model(model, out_dir / "final.safetensors", game="chess")
    total_time = (time.time() - t0) / 60
    print(
        f"\nDone. {total_time:.1f} min for {step} steps. "
        f"Final loss {last_loss:.4f}, best {best_loss:.4f}. "
        f"Output: {out_dir}/final.safetensors",
        flush=True,
    )


if __name__ == "__main__":
    main()
