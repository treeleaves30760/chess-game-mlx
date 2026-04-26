"""dlshogi teacher: generate (position, policy, value, moves_left) labels via dlshogi.

Mirrors :mod:`training.data.lc0_teacher` but for SHOGI distillation. Wraps the
``dlshogi`` (or ``DeepLearningShogi``) USI binary, asks it to evaluate random
positions with MCTS, and persists soft-policy labels for KL distillation.

**Why the MultiPV path instead of dlshogi's `info string visit=...` debug output?**
dlshogi's per-move stats line format has shifted between releases (wcsc31 vs
dr2_exhi vs wcsc32). The standard USI ``MultiPV`` option, on the other hand,
is stable across versions and trivially parseable: dlshogi emits one
``info ... multipv K score cp X pv MOVE...`` line per top-K PV.

We then convert the K cp scores to a soft probability distribution via
softmax, giving a smooth target for KL distillation. This isn't dlshogi's
exact MCTS visit-count distribution — but it captures relative move
preference and works across every dlshogi release.

**Output layout per position**:
    x:           float32 [81, 90]     (our 90-plane shogi encoding)
    policy_idx:  int64                (argmax of dlshogi multipv 1; or -1 to skip)
    value:       float32              (sente-relative; in [-1, +1])
    moves_left:  float32              (rough estimate; dlshogi has no MLH head)
    p_top_idx:   int32  [N, K_SOFT_TOP]  (-1 sentinel for pad)
    p_top_prob:  float32 [N, K_SOFT_TOP] (0.0 sentinel for pad)

**Performance** (M3 Air, dlshogi dr2_exhi model, single thread):
  ~5-8 positions/sec at MultiPV=32, nodes=64 (top-K + small MCTS)
  ~1-2 positions/sec at MultiPV=32, nodes=400 (sharper distributions)

**Licensing reminder**:
The dr2_exhi model has a *restrictive* license that explicitly permits using
it as a teacher to generate game/position labels for learning your own model.
This module is the license-compliant entry point for that use. Do NOT
redistribute the model file or commit it to git. See ``docs/dlshogi_setup.md``.
"""

from __future__ import annotations

import math
import os
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Soft-policy width — matches LC0 teacher convention so downstream training
# scripts can consume either dataset shape interchangeably.
K_SOFT_TOP = 32

try:
    import cshogi  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover - cshogi is required for shogi
    raise RuntimeError(
        "cshogi is required for the dlshogi teacher. Install via:\n"
        "  uv add 'cshogi @ git+https://github.com/TadaoYamaoka/cshogi'"
    ) from e

from training.data.encoding import (
    encode_shogi_position,
    shogi_move_to_idx,
)


# Regex for parsing dlshogi `info ... multipv K ... score cp X ... pv MOVE ...`
# Example line:
#   info depth 2 seldepth 2 multipv 1 score cp 35 nodes 64 nps 320 time 200 pv 7g7f 8c8d
_MULTIPV_LINE = re.compile(
    r"^info\b.*?\bmultipv\s+(\d+)\b.*?\bscore\s+cp\s+(-?\d+)\b.*?\bpv\s+(\S+)"
)
# dlshogi may also emit mate scores as "score mate N" — treat as ±extreme cp.
_MULTIPV_MATE_LINE = re.compile(
    r"^info\b.*?\bmultipv\s+(\d+)\b.*?\bscore\s+mate\s+(-?\d+)\b.*?\bpv\s+(\S+)"
)


def _cp_to_value(cp: int) -> float:
    """Convert centipawn score to a [-1, +1] value via the standard
    AlphaZero-style mapping (sigmoid on log-odds).
    """
    # cp -> win-rate: WR = 1 / (1 + 10^(-cp/400))
    # value = 2*WR - 1, in [-1, +1]
    wr = 1.0 / (1.0 + math.pow(10.0, -cp / 400.0))
    return max(-1.0, min(1.0, 2.0 * wr - 1.0))


def _softmax_probs(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over a list of float scores."""
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / max(temperature, 1e-6)) for s in scores]
    z = sum(exps)
    if z <= 0.0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / z for e in exps]


@dataclass
class DlshogiAnnotation:
    sfen: str
    policy_idx: int
    value: float                  # sente-relative; [-1, +1]
    moves_left: float             # rough constant since dlshogi has no MLH head
    soft_top_idx: list[int] = field(default_factory=list)
    soft_top_prob: list[float] = field(default_factory=list)


class DlshogiTeacher:
    """Wraps a dlshogi USI subprocess for synchronous per-position annotation.

    Uses USI ``MultiPV`` so it works across every dlshogi release without
    relying on version-specific ``info string`` debug formats.
    """

    def __init__(
        self,
        engine_path: str | Path,
        model_path: str | Path,
        multipv: int = K_SOFT_TOP,
        nodes: int = 64,
        threads: int = 1,
        warmup_timeout: float = 120.0,
        per_position_timeout: float = 15.0,
        softmax_temperature: float = 1.5,
        extra_setoption: list[tuple[str, str]] | None = None,
    ):
        """Start a dlshogi subprocess with the given model loaded.

        Args:
            engine_path: Path to dlshogi USI engine binary
                (e.g. ``./bin/usi.exe`` for Windows builds, or a custom Mac
                build). Must be executable.
            model_path:  Path to the ONNX model (e.g.
                ``data/dlshogi_nets/model-dr2_exhi.onnx``).
            multipv:     How many top moves to capture per position (≤ K_SOFT_TOP).
            nodes:       MCTS playouts per position. ≥multipv recommended so
                each PV slot has at least one search.
            threads:     Threads per dlshogi subprocess. 1 = deterministic.
            softmax_temperature:
                Temperature applied to multipv cp scores when converting to
                soft probabilities. Higher = smoother distribution; 1.0 = direct
                Boltzmann; 1.5 (default) loosens dlshogi's typically-sharp
                values to give the student more learning signal.
            extra_setoption: Additional ``(name, value)`` pairs forwarded as
                ``setoption name X value Y``. Useful for engine-specific knobs
                (e.g. ``("DNN_Batch_Size", "1")``).
        """
        self.engine_path = str(engine_path)
        self.model_path = str(model_path)
        self.multipv = max(1, min(multipv, K_SOFT_TOP))
        self.nodes = max(self.multipv, nodes)
        self.threads = max(1, threads)
        self.warmup_timeout = warmup_timeout
        self.per_position_timeout = per_position_timeout
        self.softmax_temperature = softmax_temperature
        self.extra_setoption = list(extra_setoption or [])

        if not os.access(self.engine_path, os.X_OK):
            raise FileNotFoundError(
                f"dlshogi engine binary not executable: {self.engine_path}\n"
                "See docs/dlshogi_setup.md for installation steps."
            )
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"dlshogi model file not found: {self.model_path}\n"
                "Download model-dr2_exhi.zip from the GitHub release and accept the license."
            )

        self._proc = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()

        self._send("usi")
        self._read_until("usiok", timeout_lines=2000)
        # dlshogi-specific options (names match upstream wiki / WCSC builds).
        # If a build doesn't recognise an option, it logs to stderr and ignores
        # — so we silently soldier on.
        self._send(f"setoption name DNN_Model value {self.model_path}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name MultiPV value {self.multipv}")
        # Disable opening book / mate searches that interfere with raw policy.
        self._send("setoption name USI_OwnBook value false")
        self._send("setoption name OwnBook value false")
        self._send("setoption name PV_Mate_Search_Threads value 0")
        self._send("setoption name Mate_Root_Search value 0")
        # Speed up: skip late-game endgame tablebase loading
        self._send("setoption name DNN_Batch_Size value 1")
        for name, value in self.extra_setoption:
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._read_until("readyok", timeout_lines=2000, timeout_seconds=self.warmup_timeout)
        # Warmup eval — first ONNX session compile can be slow.
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
        timeout_lines: int = 5000,
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
            f"dlshogi didn't emit '{needle}' in {timeout_lines} lines / "
            f"{timeout_seconds}s; last 3: {out[-3:] if out else '<none>'}"
        )

    def _warmup(self) -> None:
        """One-time first eval to pay the ONNX session compile cost."""
        self._send("position startpos")
        self._send(f"go nodes {self.nodes}")
        self._read_until(
            "bestmove",
            timeout_lines=20000,
            timeout_seconds=self.warmup_timeout,
        )

    # ------------------------------------------------------------------
    # Per-position annotation
    # ------------------------------------------------------------------

    def annotate(self, board: "cshogi.Board") -> DlshogiAnnotation | None:
        """Return an annotated training target for ``board``. None on failure."""
        sfen = board.sfen()
        with self._lock:
            self._send(f"position sfen {sfen}")
            self._send(f"go nodes {self.nodes}")
            try:
                lines = self._read_until(
                    "bestmove",
                    timeout_lines=10000,
                    timeout_seconds=self.per_position_timeout,
                )
            except RuntimeError:
                # Recovery: stop + isready to resync
                try:
                    self._send("stop")
                    self._send("isready")
                    self._read_until("readyok", timeout_lines=500, timeout_seconds=5.0)
                except Exception:
                    pass
                return None

        return self._parse(lines, board)

    def _parse(
        self, lines: list[str], board: "cshogi.Board"
    ) -> DlshogiAnnotation | None:
        """Extract argmax + soft-policy from dlshogi's MultiPV info lines."""
        # Keep only the LAST seen line for each multipv slot — dlshogi may
        # emit successively-deeper info lines and we want the deepest.
        per_slot: dict[int, tuple[int, str]] = {}  # slot -> (cp, first-pv-move)
        best_uci: str | None = None
        sente_to_move = board.turn == cshogi.BLACK  # cshogi.BLACK = 先手

        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "resign" and parts[1] != "win":
                    best_uci = parts[1]
                continue
            m = _MULTIPV_LINE.match(line)
            if m:
                slot = int(m.group(1))
                cp = int(m.group(2))
                first_move = m.group(3)
                per_slot[slot] = (cp, first_move)
                continue
            m_mate = _MULTIPV_MATE_LINE.match(line)
            if m_mate:
                slot = int(m_mate.group(1))
                mate_in = int(m_mate.group(2))
                # Convert mate distance to extreme cp: mate-in-1 → +30000, mate-in-N → +30000-N
                cp = (30000 - abs(mate_in)) if mate_in > 0 else -(30000 - abs(mate_in))
                first_move = m_mate.group(3)
                per_slot[slot] = (cp, first_move)

        if best_uci is None and not per_slot:
            return None

        # Argmax move = multipv slot 1 (or fall back to bestmove)
        if 1 in per_slot:
            argmax_uci = per_slot[1][1]
            argmax_cp = per_slot[1][0]
        else:
            argmax_uci = best_uci or ""
            argmax_cp = 0

        # Convert cp to value (always sente-relative for our convention).
        # dlshogi reports cp from side-to-move POV — we flip to sente.
        value = _cp_to_value(argmax_cp if sente_to_move else -argmax_cp)

        # Map argmax USI to our 2187-slot index
        argmax_idx = self._usi_to_idx(argmax_uci, board)
        if argmax_idx < 0:
            return None

        # Soft policy: rank slots, take cp -> softmax probabilities
        sorted_slots = sorted(per_slot.items(), key=lambda kv: kv[0])  # by slot id
        cps = [cp for _slot, (cp, _mv) in sorted_slots]
        probs = _softmax_probs([float(c) for c in cps], temperature=self.softmax_temperature)
        soft_idx: list[int] = []
        soft_prob: list[float] = []
        for ((_slot, (_cp, mv)), p) in zip(sorted_slots, probs, strict=False):
            if len(soft_idx) >= K_SOFT_TOP:
                break
            idx = self._usi_to_idx(mv, board)
            if idx < 0:
                continue
            soft_idx.append(idx)
            soft_prob.append(float(p))

        if not soft_idx:
            soft_idx = [argmax_idx]
            soft_prob = [1.0]
        else:
            # Re-normalise after dropping any unmappable moves
            s = sum(soft_prob)
            if s > 0:
                soft_prob = [p / s for p in soft_prob]

        return DlshogiAnnotation(
            sfen=board.sfen(),
            policy_idx=argmax_idx,
            value=float(value),
            moves_left=120.0,  # rough constant; dlshogi has no MLH head
            soft_top_idx=soft_idx,
            soft_top_prob=soft_prob,
        )

    def _usi_to_idx(self, usi: str, board: "cshogi.Board") -> int:
        """Map a USI move string to our 2187-slot policy index, -1 if unmappable."""
        if not usi:
            return -1
        try:
            move = board.move_from_usi(usi)  # instance method, not module fn
            if move == 0:  # cshogi sentinel for invalid
                return -1
            return shogi_move_to_idx(move, board)
        except Exception:
            return -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self) -> "DlshogiTeacher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Random shogi position sampling (mirror of LC0 teacher's helper)
# --------------------------------------------------------------------------


def _random_shogi_position(rng: random.Random) -> "cshogi.Board":
    """Play 0..40 random legal moves from startpos."""
    board = cshogi.Board()
    n = rng.randint(0, 40)
    for _ in range(n):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


# --------------------------------------------------------------------------
# Dataset writer (mirror of save_prelabeled_dataset_lc0)
# --------------------------------------------------------------------------


def save_prelabeled_dataset_dlshogi(
    output_path: Path,
    teacher: DlshogiTeacher,
    num_positions: int,
    seed: int = 0,
    log_every: int = 100,
    checkpoint_every: int = 2000,
) -> None:
    """Generate and save dlshogi-annotated shogi positions to a .npz file.

    Schema is parallel to ``save_prelabeled_dataset_lc0`` so the same KL
    distillation training script works for either game.

    Args:
        checkpoint_every: Save partial ``.npz`` every N annotated positions
            so a crash or early exit still preserves progress. 0 disables.
    """
    rng = random.Random(seed)
    X: list[np.ndarray] = []
    P: list[int] = []
    V: list[float] = []
    M: list[float] = []
    P_TOP_IDX: list[np.ndarray] = []
    P_TOP_PROB: list[np.ndarray] = []

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
    while done < num_positions:
        attempts += 1
        if attempts > num_positions * 5:
            raise RuntimeError(
                f"Couldn't generate {num_positions} valid positions after "
                f"{attempts} attempts (got {done})"
            )
        board = _random_shogi_position(rng)
        if board.is_game_over():
            continue
        ann = teacher.annotate(board)
        if ann is None:
            continue
        X.append(encode_shogi_position(board.sfen()))
        P.append(ann.policy_idx)
        V.append(ann.value)
        M.append(ann.moves_left)
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
            eta_min = (num_positions - done) / max(rate, 1e-6) / 60.0
            print(
                f"  annotated {done}/{num_positions}  "
                f"({rate:.1f}/s, eta {eta_min:.1f} min)",
                flush=True,
            )
        if checkpoint_every > 0 and done % checkpoint_every == 0:
            _save_snapshot(output_path)
            print(f"  [checkpoint] wrote {done} positions", flush=True)

    _save_snapshot(output_path)
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(
        f"saved {done} positions to {output_path} "
        f"({size_mb:.1f} MB, {elapsed/60:.1f} min, {done/elapsed:.1f} pos/s)"
    )
