"""
Convert floodgate CSA shogi game records to dlshogi-format ``.hcpe`` files.

Usage::

    uv run python training/scripts/csa_to_hcpe.py \\
        --csa-dir data/dlshogi/floodgate_2025_raw \\
        --output  data/dlshogi/floodgate_2025_R2700.hcpe \\
        --min-rating 2700 \\
        --min-moves 30 \\
        --skip-jishogi

Filter heuristics (similar to dlshogi's csa_to_hcpe.py defaults):
  * Both players' Elo (CSA "'black_rate" / "'white_rate") must be ≥ ``min_rating``.
  * Game must reach at least ``min_moves`` plies.
  * If ``--skip-jishogi`` is set, drop games that ended by mutual entering king
    (持将棋), repetition, or unfinished session.
  * Drop the position immediately preceding mate (``%TSUMI``) — dlshogi
    excludes these because the value head trains poorly on lost-by-mate.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import numpy as np


# CSA endgame string → "wins?" / "playable for hcpe?" classification
_ENDGAME_KEEP = {
    "%TORYO",        # resignation — keep all moves
    "%KACHI",        # entering-king win
    "%TSUMI",        # mate
    "%FUZUMI",       # checkmated (synonym in some servers)
    "%ILLEGAL_MOVE",  # rare; the loser's last move was illegal — keep the rest
}
_ENDGAME_DROP = {
    "%CHUDAN",        # disconnect mid-game
    "%SENNICHITE",    # repetition draw — value signal is ambiguous
    "%JISHOGI",       # mutual entering king
    "%MAX_MOVES",     # exceeded ply cap
    "%TIME_UP",       # one side timed out — eval is unreliable
    "%ERROR",
    "",               # unknown / corrupt
}


def _iter_csa_files(root: Path):
    if root.is_file():
        yield root
        return
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".csa"):
                yield Path(dirpath) / fn


@click.command()
@click.option("--csa-dir", type=str, required=True, help="Directory or single .csa file.")
@click.option("--output", type=str, required=True, help="Output .hcpe path.")
@click.option("--min-rating", type=int, default=2700, show_default=True,
              help="Minimum Elo rating for both players.")
@click.option("--min-moves", type=int, default=30, show_default=True,
              help="Skip games with fewer than this many plies.")
@click.option("--skip-jishogi/--keep-jishogi", default=True, show_default=True,
              help="Drop games ending in mutual entering-king / draws.")
@click.option("--max-games", type=int, default=None,
              help="Hard limit on number of games to process (for quick runs).")
@click.option("--report-every", type=int, default=2000, show_default=True)
def main(
    csa_dir: str,
    output: str,
    min_rating: int,
    min_moves: int,
    skip_jishogi: bool,
    max_games: int | None,
    report_every: int,
) -> None:
    """Convert filtered CSA games to a single ``.hcpe`` file."""
    import cshogi  # noqa: PLC0415
    from cshogi import CSA  # noqa: PLC0415

    csa_root = Path(csa_dir)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parser = CSA.Parser()
    hcpe_records: list[np.ndarray] = []

    games_seen = 0
    games_kept = 0
    games_skip_rating = 0
    games_skip_short = 0
    games_skip_endgame = 0
    games_skip_parse = 0

    cb = cshogi.Board()

    for csa_path in _iter_csa_files(csa_root):
        games_seen += 1
        if max_games is not None and games_seen > max_games:
            games_seen = max_games
            break

        try:
            parser.parse_csa_file(str(csa_path))
        except Exception:
            games_skip_parse += 1
            if games_seen % report_every == 0:
                _log(games_seen, games_kept, games_skip_rating, games_skip_short,
                     games_skip_endgame, games_skip_parse, len(hcpe_records))
            continue

        # Rating filter
        try:
            ratings = parser.ratings
            if not ratings or min(ratings) < min_rating:
                games_skip_rating += 1
                if games_seen % report_every == 0:
                    _log(games_seen, games_kept, games_skip_rating, games_skip_short,
                         games_skip_endgame, games_skip_parse, len(hcpe_records))
                continue
        except Exception:
            games_skip_rating += 1
            continue

        moves = list(parser.moves)
        if len(moves) < min_moves:
            games_skip_short += 1
            if games_seen % report_every == 0:
                _log(games_seen, games_kept, games_skip_rating, games_skip_short,
                     games_skip_endgame, games_skip_parse, len(hcpe_records))
            continue

        endgame = (parser.endgame or "").upper()
        if skip_jishogi and (endgame in _ENDGAME_DROP or endgame not in _ENDGAME_KEEP):
            games_skip_endgame += 1
            if games_seen % report_every == 0:
                _log(games_seen, games_kept, games_skip_rating, games_skip_short,
                     games_skip_endgame, games_skip_parse, len(hcpe_records))
            continue

        # Determine game result (cshogi enum)
        win_code = parser.win  # 0=draw/unknown, 1=BLACK_WIN, 2=WHITE_WIN per cshogi
        if win_code == cshogi.BLACK_WIN:
            game_result = cshogi.BLACK_WIN
        elif win_code == cshogi.WHITE_WIN:
            game_result = cshogi.WHITE_WIN
        else:
            game_result = 0  # treat unknown as draw

        # Per-move scores (centipawn). May be empty.
        scores = list(parser.scores) if parser.scores else []

        # Reset board to start position
        try:
            cb.set_sfen(parser.sfen)
        except Exception:
            games_skip_parse += 1
            continue

        # Drop the LAST move (often the mating move where the resulting
        # position is terminal — its hcp is still valid but bestMove leads
        # straight to mate and skews the value head).
        keep_moves = moves[:-1]

        game_records: list[np.ndarray] = []
        ok = True
        for ply, mv in enumerate(keep_moves):
            if not cb.is_legal(mv):
                ok = False
                break

            hcp = np.zeros(32, dtype=np.uint8)
            cb.to_hcp(hcp)
            entry = np.zeros(1, dtype=cshogi.HuffmanCodedPosAndEval)
            entry[0]["hcp"][:] = hcp
            entry[0]["bestMove16"] = cshogi.move16(mv)
            entry[0]["gameResult"] = game_result
            if ply < len(scores):
                entry[0]["eval"] = max(-30000, min(30000, int(scores[ply])))
            else:
                entry[0]["eval"] = 0
            game_records.append(entry)
            cb.push(mv)

        if not ok or not game_records:
            games_skip_parse += 1
            continue

        hcpe_records.extend(game_records)
        games_kept += 1

        if games_seen % report_every == 0:
            _log(games_seen, games_kept, games_skip_rating, games_skip_short,
                 games_skip_endgame, games_skip_parse, len(hcpe_records))

    if not hcpe_records:
        click.echo("No games kept after filtering — check --min-rating / --min-moves.", err=True)
        raise SystemExit(1)

    arr = np.concatenate(hcpe_records, axis=0)
    arr.tofile(str(out_path))

    click.echo("=" * 60)
    click.echo(f"Final: {games_seen:,} games scanned")
    click.echo(f"  kept            : {games_kept:,}")
    click.echo(f"  skip_rating     : {games_skip_rating:,}")
    click.echo(f"  skip_short      : {games_skip_short:,}")
    click.echo(f"  skip_endgame    : {games_skip_endgame:,}")
    click.echo(f"  skip_parse      : {games_skip_parse:,}")
    click.echo(f"  positions kept  : {len(arr):,}")
    click.echo(f"  bytes written   : {out_path.stat().st_size:,}")
    click.echo(f"  output          : {out_path}")


def _log(seen: int, kept: int, sr: int, ss: int, se: int, sp: int, pos: int) -> None:
    click.echo(
        f"  scanned={seen:,} kept={kept:,} "
        f"(skip rating={sr:,} short={ss:,} endgame={se:,} parse={sp:,}) "
        f"positions={pos:,}"
    )


if __name__ == "__main__":
    main()
