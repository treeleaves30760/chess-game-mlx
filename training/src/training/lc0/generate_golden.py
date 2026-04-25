"""Generate golden-data JSON for the C++ engine agent to validate against.

Produces two sibling files under ``shared/golden_data/``:

* ``lc0_bt4_inputs.json`` — the 10 test FENs together with their 112-plane
  input tensors (as flat float32 lists of 7168 entries each, NCHW-flattened).
* ``lc0_bt4_outputs.json`` — for each FEN, the expected post-forward results:
  * ``policy_1858`` — 1858-element float32 policy *logits* (exactly what the
    ONNX network emits at the ``/output/policy`` node)
  * ``wdl`` — 3-element softmax-normalized WDL distribution
  * ``moves_left`` — scalar
  * ``top5_legal_priors`` — top-5 legal moves by softmax-over-legal probability

These files constitute the CONTRACT between the Python reference
implementation and the C++ backend. The C++ side is correct iff it
reproduces each output array within a tight float32 tolerance (1e-4
relative on policy logits, 5% relative on normalized probabilities).

Run:

    uv run python -m training.lc0.generate_golden \
        --onnx /tmp/lc0_onnx/BT4.onnx \
        --out-inputs shared/golden_data/lc0_bt4_inputs.json \
        --out-outputs shared/golden_data/lc0_bt4_outputs.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np

from training.lc0.encoding import encode_fen
from training.lc0.move_index import move_to_nn_index
from training.lc0.onnx_runner import LC0OnnxRunner

# Ten diverse positions covering the decision-making envelope of a
# chess engine. Covers: startpos, openings (white + black POV),
# midgame tactics, endgames, and a few stress cases.
DEFAULT_FENS: list[tuple[str, str]] = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    (
        "after-e4",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    ),
    (
        "after-e4-c5",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    ),
    (
        "italian-game",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    ),
    (
        "queens-gambit-declined",
        "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3",
    ),
    (
        "kiwipete",  # classic endgame/tactic test position
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    (
        "krk-endgame",
        "8/8/8/4k3/8/4K3/8/R7 w - - 0 1",
    ),
    (
        "pawn-endgame",
        "8/5kp1/p7/P6p/4P3/7P/3K2P1/8 w - - 0 1",
    ),
    (
        "black-midgame",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 b - - 4 8",
    ),
    (
        "knight-promotion-likely",  # tactical puzzle: black has promotion threat
        "8/P4k2/8/8/8/8/5K2/8 w - - 0 1",
    ),
]


@dataclass
class GoldenInput:
    name: str
    fen: str
    is_black_to_move: bool
    planes_shape: list[int]  # [112, 8, 8]
    planes_flat: list[float]  # 7168 floats, NCHW row-major


@dataclass
class GoldenOutput:
    name: str
    fen: str
    policy_logits: list[float]  # 1858 raw logits
    policy_probs: list[float]  # 1858 softmax over full policy vector
    wdl: list[float]  # 3 softmax probs
    moves_left: float
    value_scalar: float  # P(win) - P(loss)
    top5_legal: list[dict[str, Any]]  # [{uci, nn_idx, prob_over_legal, raw_logit}, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the ONNX network exported by `lc0 leela2onnx`.",
    )
    parser.add_argument(
        "--out-inputs",
        required=True,
        help="Output path for the inputs JSON (the test-vector inputs).",
    )
    parser.add_argument(
        "--out-outputs",
        required=True,
        help="Output path for the outputs JSON (the expected forward-pass results).",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "cpu", "coreml"],
        help="ONNX Runtime execution provider. 'cpu' is deterministic, "
        "'coreml' is faster on Apple Silicon but compiles slowly on first run.",
    )
    parser.add_argument(
        "--fens-file",
        default=None,
        help="Optional JSON file with [{name, fen}] entries replacing the default list.",
    )
    args = parser.parse_args()

    if args.fens_file:
        fen_list = [
            (entry["name"], entry["fen"])
            for entry in json.loads(Path(args.fens_file).read_text())
        ]
    else:
        fen_list = DEFAULT_FENS

    providers: list | None = None
    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "coreml":
        providers = [
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}),
            "CPUExecutionProvider",
        ]
    runner = LC0OnnxRunner(args.onnx, providers=providers)
    print(f"Loaded {args.onnx} (provider={runner.provider_in_use})")

    inputs: list[GoldenInput] = []
    outputs: list[GoldenOutput] = []
    for name, fen in fen_list:
        print(f"-- {name} -- {fen}")
        board = chess.Board(fen)
        is_black = board.turn == chess.BLACK
        enc = encode_fen(fen)
        planes = enc.planes  # [112, 8, 8] float32
        inputs.append(
            GoldenInput(
                name=name,
                fen=fen,
                is_black_to_move=is_black,
                planes_shape=list(planes.shape),
                planes_flat=planes.astype(np.float32).flatten().tolist(),
            )
        )

        # Forward
        batch = planes[np.newaxis]
        raw = runner.forward_batch(batch)
        policy_logits = raw["policy"][0]  # [1858]
        wdl_softmax = _softmax(raw["wdl"][0])
        mlh = float(raw["mlh"][0, 0])

        # Compute legal softmax for sanity
        legal_idxs: list[int] = []
        legal_logits: list[float] = []
        legal_moves: list[chess.Move] = []
        for m in board.legal_moves:
            idx = move_to_nn_index(m, is_black)
            if idx is None:
                continue
            legal_idxs.append(idx)
            legal_logits.append(float(policy_logits[idx]))
            legal_moves.append(m)
        legal_probs = _softmax(np.asarray(legal_logits, dtype=np.float32)).tolist()

        # Top-5 by probability
        top_order = sorted(
            range(len(legal_moves)),
            key=lambda i: -legal_probs[i],
        )[:5]
        top5 = [
            {
                "uci": legal_moves[i].uci(),
                "nn_idx": int(legal_idxs[i]),
                "prob_over_legal": float(legal_probs[i]),
                "raw_logit": float(legal_logits[i]),
            }
            for i in top_order
        ]

        value_scalar = float(wdl_softmax[0] - wdl_softmax[2])
        outputs.append(
            GoldenOutput(
                name=name,
                fen=fen,
                policy_logits=policy_logits.astype(np.float32).tolist(),
                policy_probs=_softmax(policy_logits).astype(np.float32).tolist(),
                wdl=wdl_softmax.astype(np.float32).tolist(),
                moves_left=mlh,
                value_scalar=value_scalar,
                top5_legal=top5,
            )
        )
        print(
            f"   wdl=[{wdl_softmax[0]:.3f},{wdl_softmax[1]:.3f},{wdl_softmax[2]:.3f}] "
            f"mlh={mlh:.2f}  top1={top5[0]['uci']}@{top5[0]['prob_over_legal']*100:.1f}%"
        )

    # Write outputs
    inputs_doc: dict[str, Any] = {
        "schema_version": 1,
        "network_source": args.onnx,
        "network_input_shape": [112, 8, 8],
        "network_input_layout": "NCHW — plane i at bit b (square idx 0..63) "
        "maps to tensor[i, b // 8, b % 8]",
        "planes_dtype": "float32",
        "encoder": "training.lc0.encoding.encode_fen",
        "cases": [asdict(inp) for inp in inputs],
    }
    outputs_doc: dict[str, Any] = {
        "schema_version": 1,
        "network_source": args.onnx,
        "policy_size": 1858,
        "policy_layout": "index into LC0's kMoveStrs[1858] (see "
        "training/src/training/lc0/lc0_move_strs.txt). Moves are oriented as if "
        "the side-to-move plays White; for black-to-move positions, mirror "
        "ranks 1..8 before looking up.",
        "wdl_layout": "[win, draw, loss] probabilities (softmaxed). "
        "'win' is from the perspective of the side to move.",
        "moves_left_semantics": "Expected number of plies (half-moves) left in the game.",
        "tolerance_policy_prob": 0.005,
        "tolerance_wdl": 0.01,
        "cases": [asdict(out) for out in outputs],
    }

    out_inputs = Path(args.out_inputs)
    out_outputs = Path(args.out_outputs)
    out_inputs.parent.mkdir(parents=True, exist_ok=True)
    out_outputs.parent.mkdir(parents=True, exist_ok=True)
    out_inputs.write_text(json.dumps(inputs_doc, separators=(",", ":")))
    out_outputs.write_text(json.dumps(outputs_doc, separators=(",", ":")))
    print(
        f"Wrote {len(inputs)} inputs → {out_inputs} ({out_inputs.stat().st_size/1e6:.1f} MB)"
    )
    print(
        f"Wrote {len(outputs)} outputs → {out_outputs} ({out_outputs.stat().st_size/1e6:.1f} MB)"
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


if __name__ == "__main__":
    main()
