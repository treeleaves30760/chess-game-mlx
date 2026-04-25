"""Label chess positions with LC0 and save as .npz for distillation training.

Usage:
    uv run python training/scripts/label_lc0.py \
        --weights data/lc0_nets/t1_256_distilled.pb.gz \
        --output data/lc0_labels_20k.npz \
        --num-positions 20000 \
        --nodes 1 --seed 42
"""

from __future__ import annotations

from pathlib import Path

import click

from training.data.lc0_teacher import LC0Teacher, save_prelabeled_dataset_lc0


@click.command()
@click.option(
    "--weights",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/lc0_nets/t1_256_distilled.pb.gz"),
    help="Path to LC0 .pb.gz network. Default: t1_256_distilled (fast).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/lc0_labels_20k.npz"),
)
@click.option("--num-positions", type=int, default=20000)
@click.option(
    "--nodes",
    type=int,
    default=1,
    help="MCTS nodes per position. 1 = raw policy/eval (fastest).",
)
@click.option("--seed", type=int, default=42)
@click.option("--lc0-path", type=str, default="lc0")
@click.option("--warmup-timeout", type=float, default=120.0)
@click.option("--per-position-timeout", type=float, default=10.0)
@click.option("--log-every", type=int, default=200)
def main(
    weights: Path,
    output: Path,
    num_positions: int,
    nodes: int,
    seed: int,
    lc0_path: str,
    warmup_timeout: float,
    per_position_timeout: float,
    log_every: int,
) -> None:
    """Label positions with LC0."""
    print(f"Starting LC0 teacher with weights={weights}")
    print(f"  nodes={nodes}, seed={seed}")
    teacher = LC0Teacher(
        weights_path=weights,
        lc0_path=lc0_path,
        nodes=nodes,
        warmup_timeout=warmup_timeout,
        per_position_timeout=per_position_timeout,
    )
    try:
        save_prelabeled_dataset_lc0(
            output_path=output,
            teacher=teacher,
            num_positions=num_positions,
            seed=seed,
            log_every=log_every,
        )
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
