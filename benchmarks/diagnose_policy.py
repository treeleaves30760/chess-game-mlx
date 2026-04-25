"""Diagnose policy distribution at startpos for a trained model.

Usage:
    uv run python benchmarks/diagnose_policy.py checkpoints/my_model/final.safetensors [model_name]
"""
from __future__ import annotations
import sys
from pathlib import Path

import chess
import mlx.core as mx
import numpy as np


def diagnose_policy(weights_path: str, model_name: str = "Model") -> dict:
    from training.export import load_for_inference
    from training.data.encoding import encode_chess_position, chess_idx_to_move

    print(f"\n=== {model_name} ===")
    print(f"Weights: {weights_path}")

    model = load_for_inference(weights_path, game="chess")

    # Encode startpos
    board = chess.Board()
    enc = encode_chess_position(board)
    x = mx.array(enc[None])  # [1, 64, 19]

    out = model(x, game="chess")
    logits = out["policy"][0]  # [1858]
    mx.eval(logits)

    # Softmax
    logits_np = np.array(logits)
    logits_np -= logits_np.max()  # numerical stability
    exp_logits = np.exp(logits_np)
    probs_np = exp_logits / exp_logits.sum()

    # Get top-20 moves
    top_idx = np.argsort(probs_np)[::-1][:20]

    print(f"\nTop-20 policy predictions at startpos:")
    print(f"{'Rank':>4}  {'Move':>6}  {'Prob':>7}  {'vs Uniform':>12}")

    legal_moves = {m.uci() for m in board.legal_moves}
    total_top3 = 0.0
    uniform = 1.0 / len(legal_moves)  # 20 legal moves = 5%

    key_moves = {"e2e4": None, "d2d4": None, "g1f3": None}
    for rank, idx in enumerate(top_idx):
        mv = chess_idx_to_move(int(idx), board)
        move_str = mv.uci() if mv is not None else f"idx={idx}"
        prob = float(probs_np[idx])
        legal_mark = "*" if move_str in legal_moves else " "
        ratio = prob / uniform
        print(f"  {rank+1:2d}  {move_str:>6}{legal_mark}  {prob:6.2%}  ({ratio:.1f}x uniform)")
        if rank < 3:
            total_top3 += prob
        if move_str in key_moves:
            key_moves[move_str] = prob

    print(f"\nKey metrics:")
    print(f"  Top-3 probability mass: {total_top3:.2%} (target: >40%)")
    print(f"  Top-1 prob/uniform: {float(probs_np[top_idx[0]])/uniform:.1f}x (target: >3x)")
    for mv, prob in key_moves.items():
        if prob is not None:
            print(f"  {mv}: {prob:.2%} (ratio: {prob/uniform:.1f}x)")
        else:
            print(f"  {mv}: not in top-20")

    # Value at startpos
    val = float(out["value"][0, 0])
    print(f"  Value at startpos: {val:.4f} (expected ~0.0)")

    return {
        "top3_mass": total_top3,
        "top1_ratio": float(probs_np[top_idx[0]]) / uniform,
        "value": val,
        "key_moves": key_moves,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python benchmarks/diagnose_policy.py <weights_path> [model_name]")
        sys.exit(1)

    weights = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else Path(weights).parent.name
    diagnose_policy(weights, name)
