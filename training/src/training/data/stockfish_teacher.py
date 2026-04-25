"""Stockfish teacher: generate annotated (position, policy, value) training batches.

Uses the installed `stockfish` binary to label random positions with:
- `value`: win probability in [-1, +1] (white-relative, via score conversion)
- `policy`: one-hot or sharpened distribution over legal moves based on stockfish top-N

This is the **"ChessBench-lite" loader** — we don't need DeepMind's 15B-position
dataset; we can generate moderate amounts of labels on-demand using Stockfish.
Expected throughput on M3 Air with stockfish depth 10: ~30-60 positions/sec.
"""

from __future__ import annotations

import math
import random
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import mlx.core as mx
import numpy as np

from training.data.encoding import (
    chess_move_to_idx,
    encode_chess_position,
)


@dataclass
class AnnotatedPosition:
    fen: str
    policy_idx: int   # top-1 move from Stockfish
    value: float       # win prob [-1, +1] from white's POV
    moves_left: float  # estimated plies to end


def _cp_to_value(cp: int) -> float:
    """Convert centipawns to [-1, +1] win probability via sigmoid.
    Using the well-known `2 * sigmoid(cp/100 * 0.5) - 1` rule."""
    return 2.0 / (1.0 + math.exp(-0.004 * cp)) - 1.0


class StockfishTeacher:
    """Wraps a Stockfish subprocess for synchronous annotation.

    Not thread-safe; create one per thread if parallelizing.
    """

    def __init__(self, stockfish_path: str = "stockfish", depth: int = 10, threads: int = 1):
        self.depth = depth
        self._proc = subprocess.Popen(
            [stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._send("uci")
        self._read_until("uciok", timeout_lines=200)
        self._send(f"setoption name Threads value {threads}")
        self._send("setoption name Hash value 64")
        self._send("isready")
        self._read_until("readyok", timeout_lines=20)

    def _send(self, cmd: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read_until(self, needle: str, timeout_lines: int = 500) -> list[str]:
        assert self._proc.stdout is not None
        out: list[str] = []
        for _ in range(timeout_lines):
            line = self._proc.stdout.readline()
            if not line:
                break
            out.append(line.rstrip())
            if line.startswith(needle):
                return out
        raise RuntimeError(f"Stockfish didn't return '{needle}' in {timeout_lines} lines. Last: {out[-5:] if out else '<none>'}")

    def annotate(self, board: chess.Board) -> AnnotatedPosition:
        """Return annotated training target for the given board position."""
        with self._lock:
            self._send("ucinewgame")
            self._send(f"position fen {board.fen()}")
            self._send(f"go depth {self.depth}")
            lines = self._read_until("bestmove")

        best_move_uci = None
        score_cp = 0
        mate = None
        for line in lines:
            if line.startswith("info") and " score " in line:
                parts = line.split()
                try:
                    si = parts.index("score")
                    kind = parts[si + 1]
                    val = int(parts[si + 2])
                    if kind == "cp":
                        score_cp = val
                        mate = None
                    elif kind == "mate":
                        mate = val
                except (ValueError, IndexError):
                    pass
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best_move_uci = parts[1]
                break

        # Flip sign if black-to-move (stockfish returns STM-relative, we want white-relative)
        stm_sign = 1 if board.turn == chess.WHITE else -1
        if mate is not None:
            # Mate in N — saturate near ±1
            value = stm_sign * (1.0 if mate > 0 else -1.0) * 0.98
        else:
            value = stm_sign * _cp_to_value(score_cp)

        # Policy target = top-1 move
        if best_move_uci is None:
            # Terminal position — policy is undefined; caller should skip
            policy_idx = -1
        else:
            mv = chess.Move.from_uci(best_move_uci)
            try:
                policy_idx = chess_move_to_idx(mv, board)
            except Exception:
                policy_idx = -1  # out-of-schema move (rare underpromotion)

        # Moves-left: rough estimate based on fullmove_number (not great but OK)
        moves_left = max(0.0, 100.0 - float(board.fullmove_number) * 2)

        return AnnotatedPosition(
            fen=board.fen(),
            policy_idx=policy_idx,
            value=value,
            moves_left=moves_left,
        )

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _random_position(rng: random.Random) -> chess.Board:
    """Play 0..40 random legal moves from startpos."""
    board = chess.Board()
    n = rng.randint(0, 40)
    for _ in range(n):
        if board.is_game_over(claim_draw=True):
            break
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def generate_batch(
    teacher: StockfishTeacher,
    batch_size: int,
    rng: random.Random | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Generate one training batch of annotated positions via Stockfish."""
    rng = rng or random.Random()

    positions: list[np.ndarray] = []
    policies: list[int] = []
    values: list[float] = []
    moves_left: list[float] = []

    tries = 0
    while len(positions) < batch_size:
        tries += 1
        if tries > batch_size * 5:
            raise RuntimeError("Failed to generate enough non-terminal positions")
        board = _random_position(rng)
        if board.is_game_over(claim_draw=True):
            continue
        ann = teacher.annotate(board)
        if ann.policy_idx < 0:
            continue
        positions.append(encode_chess_position(board))
        policies.append(ann.policy_idx)
        values.append(ann.value)
        moves_left.append(ann.moves_left)

    x = mx.array(np.stack(positions), dtype=mx.float32)
    p = mx.array(np.asarray(policies, dtype=np.int64))
    v = mx.array(np.asarray(values, dtype=np.float32))
    m = mx.array(np.asarray(moves_left, dtype=np.float32))
    return x, p, v, m


def stream_batches(
    teacher: StockfishTeacher,
    batch_size: int,
    seed: int = 0,
) -> Iterator[tuple[mx.array, mx.array, mx.array, mx.array]]:
    """Infinite stream of training batches. Safe to wrap in a background thread."""
    rng = random.Random(seed)
    while True:
        yield generate_batch(teacher, batch_size, rng)


def save_prelabeled_dataset(
    out_path: Path,
    teacher: StockfishTeacher,
    num_positions: int,
    seed: int = 0,
) -> None:
    """Generate and save a batch of labeled positions to disk as .npz for reuse.

    Useful to amortize stockfish annotation cost — generate once, train many epochs.
    """
    rng = random.Random(seed)
    X = []
    P = []
    V = []
    M = []
    done = 0
    while done < num_positions:
        board = _random_position(rng)
        if board.is_game_over(claim_draw=True):
            continue
        ann = teacher.annotate(board)
        if ann.policy_idx < 0:
            continue
        X.append(encode_chess_position(board))
        P.append(ann.policy_idx)
        V.append(ann.value)
        M.append(ann.moves_left)
        done += 1
        if done % 50 == 0:
            print(f"  annotated {done}/{num_positions}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=np.stack(X).astype(np.float32),
        p=np.asarray(P, dtype=np.int64),
        v=np.asarray(V, dtype=np.float32),
        m=np.asarray(M, dtype=np.float32),
    )
    print(f"saved {done} positions to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def load_prelabeled_dataset(path: Path, batch_size: int, shuffle: bool = True, seed: int = 0):
    """Infinite iterator of mini-batches from a pre-labelled .npz dataset."""
    data = np.load(path)
    x, p, v, m = data["x"], data["p"], data["v"], data["m"]
    n = len(x)
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n - batch_size + 1, batch_size):
            b = idx[s:s + batch_size]
            yield (
                mx.array(x[b]),
                mx.array(p[b]),
                mx.array(v[b]),
                mx.array(m[b]),
            )
