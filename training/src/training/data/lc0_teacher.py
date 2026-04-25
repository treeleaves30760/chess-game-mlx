"""LC0 teacher: generate (position, policy, value, moves_left) training labels via LC0.

This wraps an ``lc0`` subprocess in UCI mode and parses its ``info string``
move-stats output to extract per-move policy priors plus the node-level
win/loss/draw estimate. Output schema matches ``stockfish_teacher.py`` so the
same ``pretrain_stockfish.py`` training script can consume either dataset.

**Why LC0 vs Stockfish as teacher?**
LC0 running even a single node produces policy *distributions* over all legal
moves derived from a ~2800-3400 Elo neural network. Stockfish depth-10 gives
you only the top-1 move and a scalar centipawn score. LC0 labels are far
richer for distillation — we get the full policy prior, not just argmax.

**Output layout per position**:
    x:           float32 [64, 19]     (our native chess encoding)
    policy_idx:  int64                (argmax of lc0 policy; top move)
    value:       float32              (position value, white's POV, in [-1, +1])
    moves_left:  float32              (lc0's moves-left head output if available)

We deliberately emit only policy_idx (not the full distribution) to keep the
.npz file format identical with the Stockfish dataset — the training loop
one-hots the single index at train time. A richer follow-up could save the
top-K moves + probabilities, but that's out of scope for this session.

**Performance notes**:
  - Startup cost is large (t1_256: ~20-30s to init Metal backend + load weights).
    BT4 can take 60+s. Use a single resident process per worker.
  - Per-position throughput after warmup on M3: ~3 positions/sec for t1_256
    at 1 node, ~0.2 positions/sec for BT4. Prefer t1_256 unless you need
    maximum label quality.
  - Some positions (usually mid-game ones with large legal-move sets) may
    cause the UCI handshake to stall beyond the usual 100 ms budget. We apply
    a per-position timeout and skip on failure, recovering with ``stop`` +
    ``isready``.

**Caveats**:
  - The returned value is *white-relative* (stm-corrected) so it is
    directly comparable to ``stockfish_teacher.value``.
  - LC0 policy percentages sum to ~100% only across legal moves already;
    illegal moves are not listed.
  - Some nets do not expose a moves-left head; we fall back to a rough
    estimate in that case.
"""

from __future__ import annotations

import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Number of top moves to persist per position in soft-policy training data.
# 32 comfortably covers every legal chess move across all normal positions
# (max legal moves is ~218 but ≥99.9% of probability mass is in the top-20).
K_SOFT_TOP = 32

import chess
import mlx.core as mx
import numpy as np

from training.data.encoding import (
    chess_move_to_idx,
    encode_chess_position,
)


# Regex for parsing LC0 `info string <move>  (idx) N: ... (P: xx.xx%) ... (Q: ...) ...`
# Example line:
#   info string d2d4  (293 ) N:       0 (+ 0) (P: 13.05%) (WL:  -.-----) (D: -.---)
_MOVE_LINE = re.compile(
    r"^info string (\S+)\s+\(\s*(\d+)\s*\)\s+N:\s*(\d+)\s+"
    r"\(\+\s*\d+\)\s+\(P:\s*([-\d.]+)%\)"
)
# Example node line (summary of the root node):
#   info string node  (  20) N: 1 (+ 0) (P:  0.00%) (WL:  0.03661) (D: 0.584) (M: 185.1)
_NODE_LINE = re.compile(
    r"^info string node\s+\(\s*\d+\s*\)\s+N:\s*\d+\s+\(\+\s*\d+\)\s+"
    r"\(P:\s*[-\d.]+%\)\s+\(WL:\s*([-\d.]+)\)\s+\(D:\s*([-\d.]+)\)\s+"
    r"\(M:\s*([-\d.]+)\)"
)


@dataclass
class LC0Annotation:
    fen: str
    policy_idx: int
    value: float                  # white-relative; [-1, +1]
    moves_left: float             # rough estimate if not provided
    # Soft policy: top-K moves by LC0 prior, as (policy_idx, probability) pairs.
    # Sorted by descending probability. May be shorter than K_SOFT_TOP if the
    # position has fewer legal moves (unlikely in practice but guarded).
    # Probabilities sum to ~1.0 across all captured entries.
    soft_top_idx: list[int]       = field(default_factory=list)
    soft_top_prob: list[float]    = field(default_factory=list)


class LC0Teacher:
    """Wraps an lc0 subprocess for synchronous per-position annotation.

    Not thread-safe; create one per thread if parallelising. On Apple Silicon
    the Metal backend already uses the GPU, so a single teacher already
    saturates the available compute.
    """

    def __init__(
        self,
        weights_path: str | Path,
        lc0_path: str = "lc0",
        nodes: int = 1,
        warmup_timeout: float = 90.0,
        per_position_timeout: float = 10.0,
    ):
        """Start an lc0 subprocess with the given weights loaded.

        Args:
            weights_path:   Path to an LC0 ``.pb.gz`` network file.
            lc0_path:       Path to the ``lc0`` binary (``lc0`` on PATH by default).
            nodes:          Number of MCTS nodes per position. 1 = raw policy/eval.
            warmup_timeout: Seconds to wait for the first search after loading
                            weights. BT4 on Metal can take 60-90 s.
            per_position_timeout:
                            Seconds to wait for each subsequent ``bestmove``.
        """
        self.weights_path = str(weights_path)
        self.nodes = nodes
        self.warmup_timeout = warmup_timeout
        self.per_position_timeout = per_position_timeout

        self._proc = subprocess.Popen(
            [lc0_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._send("uci")
        self._read_until("uciok", timeout_lines=1000)
        # Essential options for labelling:
        self._send(f"setoption name WeightsFile value {self.weights_path}")
        self._send("setoption name VerboseMoveStats value true")
        # Prefer a fast backend and 1-node searches (skip MCTS expansion)
        self._send("setoption name MinibatchSize value 1")
        # Deterministic behaviour:
        self._send("setoption name Temperature value 0")
        self._send("isready")
        self._read_until("readyok", timeout_lines=500)
        # Warmup: first eval of a fresh network is slow (Metal shader compile
        # + weight upload), so do it once here with a big timeout.
        self._warmup()

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def _send(self, cmd: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read_until(
        self,
        needle: str,
        timeout_lines: int = 3000,
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """Read stdout lines until one starts with ``needle``. Raises on timeout."""
        assert self._proc.stdout is not None
        out: list[str] = []
        t0 = time.time()
        for _ in range(timeout_lines):
            if timeout_seconds is not None and time.time() - t0 > timeout_seconds:
                break
            line = self._proc.stdout.readline()
            if not line:
                break
            out.append(line.rstrip())
            if line.startswith(needle):
                return out
        raise RuntimeError(
            f"lc0 didn't emit '{needle}' in {timeout_lines} lines / "
            f"{timeout_seconds}s; last 3: {out[-3:] if out else '<none>'}"
        )

    def _warmup(self) -> None:
        """One-time first eval to pay the startup cost."""
        t0 = time.time()
        self._send("position startpos")
        self._send(f"go nodes {self.nodes}")
        # Larger timeout because Metal init may not be complete yet
        self._read_until(
            "bestmove",
            timeout_lines=20000,
            timeout_seconds=self.warmup_timeout,
        )

    # ------------------------------------------------------------------
    # Per-position annotation
    # ------------------------------------------------------------------

    def annotate(self, board: chess.Board) -> LC0Annotation | None:
        """Return an annotated training target for ``board``. None on failure."""
        fen = board.fen()
        with self._lock:
            self._send(f"position fen {fen}")
            self._send(f"go nodes {self.nodes}")
            try:
                lines = self._read_until(
                    "bestmove",
                    timeout_lines=5000,
                    timeout_seconds=self.per_position_timeout,
                )
            except RuntimeError:
                # Recovery: send stop + isready to resync
                try:
                    self._send("stop")
                    self._send("isready")
                    self._read_until(
                        "readyok",
                        timeout_lines=500,
                        timeout_seconds=5.0,
                    )
                except Exception:
                    pass
                return None

        # Parse the output
        return self._parse(lines, board)

    @staticmethod
    def _parse(lines: list[str], board: chess.Board) -> LC0Annotation | None:
        """Parse LC0's info strings into an annotation with soft policy."""
        best_uci: str | None = None
        wl: float = 0.0
        moves_left: float = 100.0
        # Collect every legal move's P-prior as we see it.
        per_move_p: list[tuple[str, float]] = []

        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best_uci = parts[1]
                continue
            # Check node line first because "node" is a valid \S+ for the move regex
            m_node = _NODE_LINE.match(line)
            if m_node:
                try:
                    wl = float(m_node.group(1))
                    moves_left = float(m_node.group(3))
                except ValueError:
                    pass
                continue
            m = _MOVE_LINE.match(line)
            if m:
                uci = m.group(1)
                try:
                    p_pct = float(m.group(4))
                except ValueError:
                    continue
                per_move_p.append((uci, p_pct))

        if best_uci is None:
            return None

        # Value — LC0 reports WL from stm POV; convert to white POV
        stm_sign = 1 if board.turn == chess.WHITE else -1
        value = max(-1.0, min(1.0, stm_sign * wl))

        # Map bestmove to policy idx
        try:
            mv = chess.Move.from_uci(best_uci)
            policy_idx = chess_move_to_idx(mv, board)
        except Exception:
            policy_idx = -1
        if policy_idx < 0:
            return None

        # Build soft policy: sort by descending P, map uci→policy_idx, take top-K,
        # normalize probabilities so they sum to 1.
        per_move_p.sort(key=lambda kv: -kv[1])
        soft_idx: list[int] = []
        soft_prob: list[float] = []
        for uci, p_pct in per_move_p:
            if len(soft_idx) >= K_SOFT_TOP:
                break
            try:
                mv2 = chess.Move.from_uci(uci)
                idx = chess_move_to_idx(mv2, board)
            except Exception:
                continue
            if idx < 0:
                continue
            soft_idx.append(idx)
            soft_prob.append(p_pct / 100.0)  # pct -> fraction

        if not soft_idx:
            # Fallback: at least record the argmax move
            soft_idx = [policy_idx]
            soft_prob = [1.0]
        else:
            s = sum(soft_prob)
            if s > 0:
                soft_prob = [p / s for p in soft_prob]

        return LC0Annotation(
            fen=board.fen(),
            policy_idx=policy_idx,
            value=float(value),
            moves_left=float(moves_left),
            soft_top_idx=soft_idx,
            soft_top_prob=soft_prob,
        )

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# --------------------------------------------------------------------------
# Position sampling (matches stockfish_teacher semantics)
# --------------------------------------------------------------------------

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


def save_prelabeled_dataset_lc0(
    output_path: Path,
    teacher: LC0Teacher,
    num_positions: int,
    seed: int = 0,
    log_every: int = 100,
    checkpoint_every: int = 2000,
) -> None:
    """Generate and save LC0-annotated positions to a .npz file.

    Writes the same layout as ``save_prelabeled_dataset`` in
    ``stockfish_teacher`` so the training pipeline is format-agnostic.

    Args:
        checkpoint_every: Save partial ``.npz`` every N annotated positions
            so a crash or early exit still preserves progress. Pass 0 to
            disable. The partial checkpoint is written to the same path
            as ``output_path``.
    """
    rng = random.Random(seed)
    X: list[np.ndarray] = []
    P: list[int] = []
    V: list[float] = []
    M: list[float] = []
    # Soft policy: padded to fixed width K_SOFT_TOP with -1/0 sentinel
    P_TOP_IDX: list[np.ndarray] = []   # int32 [N, K_SOFT_TOP]; -1 = pad
    P_TOP_PROB: list[np.ndarray] = []  # float32 [N, K_SOFT_TOP]; 0.0 on pad

    def _save_snapshot(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=np.stack(X).astype(np.float32),
            p=np.asarray(P, dtype=np.int64),
            v=np.asarray(V, dtype=np.float32),
            m=np.asarray(M, dtype=np.float32),
            p_top_idx=np.stack(P_TOP_IDX).astype(np.int32),
            p_top_prob=np.stack(P_TOP_PROB).astype(np.float32),
        )

    t0 = time.time()
    done = 0
    attempts = 0
    try:
        while done < num_positions:
            attempts += 1
            if attempts > num_positions * 5:
                raise RuntimeError(
                    f"Couldn't generate {num_positions} valid positions after "
                    f"{attempts} attempts (got {done})"
                )
            board = _random_position(rng)
            if board.is_game_over(claim_draw=True):
                continue
            ann = teacher.annotate(board)
            if ann is None:
                continue
            X.append(encode_chess_position(board))
            P.append(ann.policy_idx)
            V.append(ann.value)
            M.append(ann.moves_left)
            # Pad soft policy to K_SOFT_TOP
            idx_arr = np.full(K_SOFT_TOP, -1, dtype=np.int32)
            prob_arr = np.zeros(K_SOFT_TOP, dtype=np.float32)
            k = min(K_SOFT_TOP, len(ann.soft_top_idx))
            idx_arr[:k] = np.asarray(ann.soft_top_idx[:k], dtype=np.int32)
            prob_arr[:k] = np.asarray(ann.soft_top_prob[:k], dtype=np.float32)
            P_TOP_IDX.append(idx_arr)
            P_TOP_PROB.append(prob_arr)
            done += 1
            if done % log_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (num_positions - done) / max(rate, 1e-6)
                print(
                    f"  annotated {done}/{num_positions}  "
                    f"({rate:.1f}/s, eta {eta/60:.1f} min)",
                    flush=True,
                )
            # Periodic checkpoint to guard against crashes
            if checkpoint_every > 0 and done % checkpoint_every == 0:
                _save_snapshot(output_path)
                print(f"  [checkpoint] wrote {done} positions", flush=True)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted at {done}/{num_positions} positions; "
            f"saving partial dataset to {output_path}",
            flush=True,
        )

    _save_snapshot(output_path)
    total_time = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6
    print(
        f"saved {done} positions to {output_path} "
        f"({size_mb:.1f} MB, {total_time/60:.1f} min, "
        f"{done/max(total_time, 1e-6):.1f} pos/s)",
        flush=True,
    )


def load_prelabeled_dataset(
    path: Path,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
):
    """Infinite iterator of mini-batches from a pre-labelled .npz dataset.

    Duplicate of ``stockfish_teacher.load_prelabeled_dataset`` for convenience;
    kept here so callers can import a single teacher module.
    """
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
