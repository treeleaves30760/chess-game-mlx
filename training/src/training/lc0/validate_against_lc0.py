"""Validate our Python ONNX path against the lc0 binary on many positions.

This script drives the lc0 binary via UCI to extract its policy priors for
each test FEN, then compares against our ONNX runner's output. It reports
the per-FEN L1 and max-deviation on the top-K legal moves.

The lc0 binary is assumed to be reachable via ``lc0`` on PATH; override
via ``--lc0-binary``.

Usage:

    uv run python -m training.lc0.validate_against_lc0 \
        --onnx /tmp/lc0_onnx/t1_256.onnx \
        --weights data/lc0_nets/t1_256_distilled.pb.gz

Success criterion: all test FENs have |prob_ours - prob_lc0| < 0.01 on
each top-10 legal move.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time

import chess

from training.lc0.onnx_runner import LC0OnnxRunner

# Lines that look like:
#   info string d2d4  (293 ) N:       1 (+ 0) (P: 23.76%) (WL:...
POLICY_LINE_RE = re.compile(
    r"^info string\s+(\S+)\s+\(\s*(\d+)\s*\)\s+.*\(P:\s*([\d.]+)%\)"
)


def query_lc0_policy(
    fen: str,
    weights: str,
    lc0_binary: str = "lc0",
    nodes: int = 2,
    timeout: float = 120.0,
) -> dict[str, tuple[int, float]]:
    """Return dict: uci_move → (nn_idx, P_percent) from lc0's VerboseMoveStats."""
    proc = subprocess.Popen(
        [lc0_binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    cmds = [
        "uci",
        f"setoption name WeightsFile value {weights}",
        # Use BLAS (CPU) backend — the Metal backend implements the promotion
        # head differently than the ONNX export (see docs). For apples-to-apples
        # comparison with our onnxruntime Python path, BLAS is the reference.
        "setoption name Backend value blas",
        "setoption name VerboseMoveStats value true",
        "setoption name SmartPruningFactor value 0.0",
        "setoption name PolicyTemperature value 1.0",
        "setoption name ContemptMode value disable",
        "setoption name Threads value 1",
        "isready",
        f"position fen {fen}" if fen != "startpos" else "position startpos",
        f"go nodes {nodes}",
    ]
    for c in cmds:
        proc.stdin.write(c + "\n")
    proc.stdin.flush()

    result: dict[str, tuple[int, float]] = {}
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            break
        m = POLICY_LINE_RE.match(line.rstrip())
        if m:
            uci_like, nn_idx_str, p_str = m.groups()
            if uci_like == "node":
                continue
            result[uci_like] = (int(nn_idx_str), float(p_str) / 100.0)
        if line.startswith("bestmove"):
            break

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return result


def compare_case(
    name: str,
    fen: str,
    runner: LC0OnnxRunner,
    weights: str,
    lc0_binary: str,
) -> dict[str, float]:
    lc0 = query_lc0_policy(fen, weights, lc0_binary=lc0_binary)
    out = runner.forward_fen(fen)

    # Build a dict of our priors keyed by the UCI move string (on the real
    # board). We intentionally skip knight-promotion moves (not in kMoveStrs).
    our_priors: dict[str, float] = {}
    for i, move in enumerate(out.legal_moves):
        uci = move.uci()
        # lc0's VerboseMoveStats reports UCI with promotion for underprom
        # but without annotation for knight-prom. Align:
        if move.promotion == chess.KNIGHT:
            uci = uci[:4]
        our_priors[uci] = float(out.legal_priors[i])

    # Compute the max deviation over the moves lc0 reported
    worst_move = ""
    worst_diff = 0.0
    sum_abs = 0.0
    n_common = 0
    for move_str, (nn_idx, p_lc0) in lc0.items():
        if move_str in our_priors:
            d = abs(our_priors[move_str] - p_lc0)
            sum_abs += d
            n_common += 1
            if d > worst_diff:
                worst_diff = d
                worst_move = move_str

    # Top-3 per side
    lc0_top = sorted(lc0.items(), key=lambda kv: -kv[1][1])[:3]
    ours_top = sorted(our_priors.items(), key=lambda kv: -kv[1])[:3]

    print(f"\n=== {name}: {fen}")
    print(f"   lc0  top3: {[(m, f'{p*100:.2f}%') for m, (_, p) in lc0_top]}")
    print(f"   ours top3: {[(m, f'{p*100:.2f}%') for m, p in ours_top]}")
    print(
        f"   n_common={n_common} mean|Δ|={sum_abs/max(n_common,1)*100:.3f}%  "
        f"max|Δ|={worst_diff*100:.3f}% (on {worst_move!r})"
    )
    return {"worst_diff": worst_diff, "mean_diff": sum_abs / max(n_common, 1)}


DEFAULT_FENS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("after-e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    ("after-e4-c5", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
    ("italian-game", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("queens-gambit", "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("krk-endgame", "8/8/8/4k3/8/4K3/8/R7 w - - 0 1"),
    ("pawn-endgame", "8/5kp1/p7/P6p/4P3/7P/3K2P1/8 w - - 0 1"),
    ("black-midgame", "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 b - - 4 8"),
    ("near-promotion", "8/P4k2/8/8/8/8/5K2/8 w - - 0 1"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--weights", required=True, help="LC0 .pb.gz weights file")
    ap.add_argument("--lc0-binary", default="lc0")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "coreml"])
    args = ap.parse_args()

    providers = (
        ["CPUExecutionProvider"]
        if args.provider == "cpu"
        else [
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}),
            "CPUExecutionProvider",
        ]
    )
    runner = LC0OnnxRunner(args.onnx, providers=providers)
    print(f"Loaded {args.onnx} (provider={runner.provider_in_use})")

    worst_overall = 0.0
    for name, fen in DEFAULT_FENS:
        r = compare_case(name, fen, runner, args.weights, args.lc0_binary)
        worst_overall = max(worst_overall, r["worst_diff"])

    print(f"\n>>> Worst deviation across all cases: {worst_overall*100:.3f}%")
    target = 0.01  # 1%
    if worst_overall < target:
        print(f">>> PASS (under {target*100:.1f}% threshold)")
    else:
        print(f">>> FAIL (exceeds {target*100:.1f}% threshold)")


if __name__ == "__main__":
    main()
