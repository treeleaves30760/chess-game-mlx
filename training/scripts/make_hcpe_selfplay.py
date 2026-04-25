"""
Generate small hcpe (HuffmanCodedPosAndEval) files from cshogi random self-play.

Used for unit-testing :class:`training.data.dlshogi_loader.DlshogiLoader` and
quickly producing a few thousand positions when no floodgate corpus is yet
available. **Not** intended as real training data — moves are uniformly random
and ``eval`` is hard-coded to 0.

Usage::

    uv run python training/scripts/make_hcpe_selfplay.py \\
        --output data/dlshogi/selfplay_random.hcpe \\
        --games 200 --max-plies 200 --seed 42
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np


@click.command()
@click.option("--output", type=str, required=True, help="Path to write .hcpe file.")
@click.option("--games", type=int, default=200, show_default=True)
@click.option("--max-plies", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(output: str, games: int, max_plies: int, seed: int) -> None:
    import cshogi  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[np.ndarray] = []

    for g in range(games):
        board = cshogi.Board()
        plies = 0
        while plies < max_plies:
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            mv = moves[rng.integers(0, len(moves))]

            # Snapshot position + chosen move BEFORE pushing
            hcp = np.zeros(32, dtype=np.uint8)
            board.to_hcp(hcp)
            entry = np.zeros(1, dtype=cshogi.HuffmanCodedPosAndEval)
            entry[0]["hcp"][:] = hcp
            entry[0]["eval"] = 0
            entry[0]["bestMove16"] = cshogi.move16(mv)
            entry[0]["gameResult"] = 0  # filled in after the game
            records.append(entry)

            board.push(mv)
            plies += 1

        # Game-over result fill
        if board.is_draw() or not board.is_game_over():
            result = 0  # DRAW or unfinished → treat as draw
        else:
            # winner is opposite side of the side-to-move at terminal
            # cshogi: BLACK=0, WHITE=1; mate ⇒ side-to-move loses
            result = cshogi.BLACK_WIN if board.turn == cshogi.WHITE else cshogi.WHITE_WIN

        # Backfill gameResult into the records produced for this game
        # (records appended in this iteration are at the tail)
        # Count of entries this game produced
        for r in records[-plies:]:
            r[0]["gameResult"] = result

        if (g + 1) % 50 == 0:
            click.echo(f"  finished {g+1}/{games} games, total positions: {len(records)}")

    if not records:
        click.echo("No positions generated (every game terminated immediately).")
        raise SystemExit(1)

    arr = np.concatenate(records, axis=0)
    arr.tofile(str(out_path))
    click.echo(f"Wrote {len(arr):,} positions to {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
