"""Label shogi positions with dlshogi and save as .npz for distillation training.

Mirror of ``label_lc0.py`` for the shogi side. Output schema is identical so
the same KL distillation training code consumes either game's labels.

Usage::

    uv run python training/scripts/label_dlshogi.py \\
        --engine /path/to/dlshogi_usi \\
        --model  data/dlshogi_nets/model-dr2_exhi.onnx \\
        --output data/dlshogi_labels_50k.npz \\
        --num-positions 50000 \\
        --multipv 32 --nodes 64 --seed 42

See ``docs/dlshogi_setup.md`` for how to install the dlshogi binary and model.
The model file is under a restrictive license — do not commit it to git or
redistribute. This script is the license-compliant entry point that uses it
purely as a teacher to generate labels for your own model training.
"""

from __future__ import annotations

from pathlib import Path

import click

from training.data.dlshogi_teacher import (
    DlshogiTeacher,
    save_prelabeled_dataset_dlshogi,
)


@click.command()
@click.option(
    "--engine",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to dlshogi USI engine binary (e.g. ./bin/dlshogi or .exe).",
)
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to dlshogi ONNX model (e.g. data/dlshogi_nets/model-dr2_exhi.onnx).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/dlshogi_labels_50k.npz"),
    show_default=True,
)
@click.option("--num-positions", type=int, default=50000, show_default=True)
@click.option(
    "--multipv",
    type=int,
    default=32,
    show_default=True,
    help="Top-K moves to capture per position (≤32, the K_SOFT_TOP constant).",
)
@click.option(
    "--nodes",
    type=int,
    default=64,
    show_default=True,
    help="MCTS playouts per position. Larger = sharper distributions but slower.",
)
@click.option(
    "--threads",
    type=int,
    default=1,
    show_default=True,
    help="dlshogi internal threads. 1 = deterministic.",
)
@click.option(
    "--softmax-temperature",
    type=float,
    default=1.5,
    show_default=True,
    help="Temperature for converting cp scores to soft probs. Higher = smoother.",
)
@click.option(
    "--warmup-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Seconds to wait for first eval (ONNX session compile can be slow).",
)
@click.option(
    "--per-position-timeout",
    type=float,
    default=15.0,
    show_default=True,
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--log-every", type=int, default=200, show_default=True)
@click.option("--checkpoint-every", type=int, default=2000, show_default=True)
def main(
    engine: Path,
    model: Path,
    output: Path,
    num_positions: int,
    multipv: int,
    nodes: int,
    threads: int,
    softmax_temperature: float,
    warmup_timeout: float,
    per_position_timeout: float,
    seed: int,
    log_every: int,
    checkpoint_every: int,
) -> None:
    """Label N random shogi positions using dlshogi as teacher; write .npz."""
    print(
        f"dlshogi labeller: engine={engine.name}, model={model.name}, "
        f"multipv={multipv}, nodes={nodes}, threads={threads}, "
        f"target={num_positions} positions"
    )
    with DlshogiTeacher(
        engine_path=engine,
        model_path=model,
        multipv=multipv,
        nodes=nodes,
        threads=threads,
        warmup_timeout=warmup_timeout,
        per_position_timeout=per_position_timeout,
        softmax_temperature=softmax_temperature,
    ) as teacher:
        save_prelabeled_dataset_dlshogi(
            output_path=output,
            teacher=teacher,
            num_positions=num_positions,
            seed=seed,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
        )


if __name__ == "__main__":
    main()
