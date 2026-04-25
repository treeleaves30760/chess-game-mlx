"""Run multiple arena matches sequentially against different Stockfish Elos.

Simple convenience wrapper that loops ``arena.py`` over a list of Elos and
aggregates the results.

Usage:
    uv run python benchmarks/arena_multi.py \
        --engine ./engine/bin/chess_engine \
        --weights checkpoints/chess_40m_lc0distill_20k/final.safetensors \
        --games 8 --movetime 1000 --elos 1320,1800,2200,2500
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


@click.command()
@click.option("--engine", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--weights", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--games", type=int, default=8)
@click.option("--movetime", type=int, default=1000)
@click.option("--opening-depth", type=int, default=6)
@click.option("--seed", type=int, default=42)
@click.option(
    "--elos",
    type=str,
    default="1320,1800,2200,2500",
    help="Comma-separated list of Stockfish UCI_Elo values to test.",
)
def main(
    engine: Path,
    weights: Path,
    games: int,
    movetime: int,
    opening_depth: int,
    seed: int,
    elos: str,
) -> None:
    """Run arena against multiple Stockfish Elos."""
    elo_list = [int(e.strip()) for e in elos.split(",") if e.strip()]
    results: list[tuple[int, str]] = []

    for elo in elo_list:
        print()
        print("=" * 60)
        print(f"Arena vs Stockfish @ UCI_Elo={elo}")
        print("=" * 60)
        cmd = [
            "uv",
            "run",
            "python",
            "benchmarks/arena.py",
            "--a",
            str(engine),
            "--a-args",
            f"--weights {weights}",
            "--b",
            "stockfish",
            "--b-setoption",
            "setoption name UCI_LimitStrength value true",
            "--b-setoption",
            f"setoption name UCI_Elo value {elo}",
            "--games",
            str(games),
            "--movetime",
            str(movetime),
            "--opening-depth",
            str(opening_depth),
            "--seed",
            str(seed + elo),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"stderr: {proc.stderr[:1000]}", file=sys.stderr)
        results.append((elo, proc.stdout))

    print()
    print("#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    for elo, output in results:
        # Extract WDL + score rate
        wdl_line = ""
        score_line = ""
        for line in output.splitlines():
            if line.startswith("Final: W-D-L"):
                wdl_line = line
            elif line.startswith("A score rate"):
                score_line = line
        print(f"  SF Elo {elo}: {wdl_line} | {score_line}")


if __name__ == "__main__":
    main()
