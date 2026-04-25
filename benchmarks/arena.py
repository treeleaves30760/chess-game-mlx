"""Simple arena: play matches between two UCI engines and report W/L/D + Elo.

Usage:
    uv run python benchmarks/arena.py \
        --a /path/to/chess_engine --a-args "--weights ckpt.safetensors" \
        --b stockfish --b-args "" \
        --games 20 --movetime 1000 --opening-depth 4

For Stockfish limited-strength:
    --b stockfish --b-args "--setoption name Skill Level value 0"

Reports:
  - games played / legal-move rate / crashes
  - W-D-L from A's perspective
  - estimated Elo diff via log(1/prob - 1) * 400
"""

from __future__ import annotations

import argparse
import math
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess


@dataclass
class EngineHandle:
    name: str
    cmd: list[str]
    proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.read_until("uciok")
        self.send("isready")
        self.read_until("readyok")

    def send(self, line: str) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, needle: str, timeout: float = 10.0) -> list[str]:
        assert self.proc is not None and self.proc.stdout is not None
        lines: list[str] = []
        end = time.time() + timeout
        while time.time() < end:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"Engine {self.name} closed pipe")
            lines.append(line.rstrip())
            if line.startswith(needle):
                return lines
        raise TimeoutError(f"Engine {self.name} didn't emit '{needle}' within {timeout}s")

    def search_bestmove(self, fen: str, moves: list[str], movetime_ms: int) -> str:
        """Return UCI bestmove string after `go movetime`."""
        if fen == chess.STARTING_FEN:
            if moves:
                self.send("position startpos moves " + " ".join(moves))
            else:
                self.send("position startpos")
        else:
            cmd = f"position fen {fen}"
            if moves:
                cmd += " moves " + " ".join(moves)
            self.send(cmd)
        self.send(f"go movetime {movetime_ms}")
        for line in self.read_until("bestmove", timeout=movetime_ms / 1000 + 10):
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1]  # bestmove e2e4 [ponder e7e5]
        raise RuntimeError("no bestmove")

    def quit(self) -> None:
        if self.proc is not None:
            try:
                self.send("quit")
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


@dataclass
class MatchStats:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    games: int = 0
    illegal_a: int = 0
    illegal_b: int = 0
    crashes_a: int = 0
    crashes_b: int = 0
    move_times_a: list[float] = field(default_factory=list)
    move_times_b: list[float] = field(default_factory=list)

    def elo_diff(self) -> float:
        if self.games == 0:
            return 0.0
        score = (self.wins_a + 0.5 * self.draws) / self.games
        if score <= 0:
            return -800.0
        if score >= 1:
            return 800.0
        return -400.0 * math.log10(1.0 / score - 1.0)


def play_one_game(
    white: EngineHandle,
    black: EngineHandle,
    movetime_ms: int,
    max_plies: int,
    opening_moves: list[str],
) -> tuple[str, list[str]]:
    """Return (result, moves). result in {'1-0', '0-1', '1/2-1/2', 'illegal_white', 'illegal_black'}."""
    board = chess.Board()
    moves_uci: list[str] = list(opening_moves)
    for m in opening_moves:
        board.push(chess.Move.from_uci(m))

    while not board.is_game_over(claim_draw=True) and len(moves_uci) < max_plies:
        to_move = white if board.turn == chess.WHITE else black
        bm = to_move.search_bestmove(chess.STARTING_FEN, moves_uci, movetime_ms)
        try:
            mv = chess.Move.from_uci(bm)
        except Exception:
            return ("illegal_white" if board.turn else "illegal_black"), moves_uci
        if mv not in board.legal_moves:
            return ("illegal_white" if board.turn else "illegal_black"), moves_uci
        board.push(mv)
        moves_uci.append(bm)

    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), moves_uci
    return "1/2-1/2", moves_uci


def random_opening(depth: int, rng: random.Random) -> list[str]:
    board = chess.Board()
    moves = []
    for _ in range(depth):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        m = rng.choice(legal)
        moves.append(m.uci())
        board.push(m)
    return moves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Engine A binary")
    ap.add_argument("--a-args", default="", help="Engine A extra CLI args (quoted)")
    ap.add_argument("--a-setoption", default=[], action="append",
                    help="e.g. 'setoption name Skill Level value 0' (repeatable)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-args", default="")
    ap.add_argument("--b-setoption", default=[], action="append")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--movetime", type=int, default=500, help="ms per move")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--opening-depth", type=int, default=4, help="random opening plies to avoid same openings")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = MatchStats()
    a_cmd = [args.a] + shlex.split(args.a_args)
    b_cmd = [args.b] + shlex.split(args.b_args)

    print(f"A: {' '.join(a_cmd)}")
    print(f"B: {' '.join(b_cmd)}")
    print(f"games={args.games} movetime={args.movetime}ms opening_depth={args.opening_depth}")
    print()

    a = EngineHandle(name="A", cmd=a_cmd)
    b = EngineHandle(name="B", cmd=b_cmd)
    try:
        a.start()
        b.start()
        for opt in args.a_setoption:
            a.send(opt)
        for opt in args.b_setoption:
            b.send(opt)

        for g in range(args.games):
            opening = random_opening(args.opening_depth, rng)
            # Alternate colours
            if g % 2 == 0:
                white, black = a, b
                a_is_white = True
            else:
                white, black = b, a
                a_is_white = False

            t0 = time.time()
            try:
                result, moves = play_one_game(white, black, args.movetime, args.max_plies, opening)
            except Exception as e:
                print(f"  game {g+1}: crash: {e}", file=sys.stderr)
                if a_is_white:
                    stats.crashes_a += 1
                else:
                    stats.crashes_b += 1
                stats.games += 1
                continue

            dur = time.time() - t0
            if result in ("illegal_white", "illegal_black"):
                side = "white" if "white" in result else "black"
                engine_label = "A" if (a_is_white and side == "white") or (not a_is_white and side == "black") else "B"
                if engine_label == "A":
                    stats.illegal_a += 1
                else:
                    stats.illegal_b += 1
                stats.games += 1
                print(f"  game {g+1}: ILLEGAL by {engine_label} at ply {len(moves)}")
                continue

            if result == "1-0":
                if a_is_white:
                    stats.wins_a += 1
                else:
                    stats.wins_b += 1
            elif result == "0-1":
                if a_is_white:
                    stats.wins_b += 1
                else:
                    stats.wins_a += 1
            else:
                stats.draws += 1
            stats.games += 1

            print(
                f"  game {g+1:>3}: {result:>8}  plies={len(moves):>3}  time={dur:5.1f}s  "
                f"running W-D-L (A) = {stats.wins_a}-{stats.draws}-{stats.wins_b}"
            )
    finally:
        a.quit()
        b.quit()

    print()
    print("=" * 60)
    print(f"Final: W-D-L (A vs B) = {stats.wins_a}-{stats.draws}-{stats.wins_b}")
    print(f"Crashes: A={stats.crashes_a} B={stats.crashes_b}")
    print(f"Illegal moves: A={stats.illegal_a} B={stats.illegal_b}")
    if stats.games > 0:
        score_a = (stats.wins_a + 0.5 * stats.draws) / stats.games
        print(f"A score rate: {score_a:.1%}")
        print(f"Estimated Elo diff (A - B): {stats.elo_diff():+.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
