"""Export a tiny random-initialised ChessShogiTransformer for tests.

The full 40 M model is too large to load in CI unit tests; this script
constructs a 4-layer / d_model=128 / 2-heads model and writes its weights
to a .safetensors file plus JSON sidecar (in the same format as
training.src.training.export.export_model).

Usage:
    uv run python -m training.scripts.export_dummy --game chess \\
            --out /tmp/dummy_chess.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

from training.export import export_model
from training.models.transformer import ChessShogiTransformer


def main() -> int:
    p = argparse.ArgumentParser(description="Export a dummy ChessShogiTransformer")
    p.add_argument("--game", choices=["chess", "shogi"], default="chess")
    p.add_argument("--out", required=True, help="Output .safetensors path")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=128)
    args = p.parse_args()

    model = ChessShogiTransformer(
        game=args.game,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
    )

    # Force weight materialisation by doing a dummy forward pass.  This
    # ensures every parameter tensor is eagerly created.
    if args.game == "chess":
        x = mx.zeros((1, 64, 19))
    else:
        x = mx.zeros((1, 81, 90))
    _ = model(x, game=args.game)

    out = Path(args.out)
    export_model(model, out, game=args.game)
    print(f"Dummy model exported: {out}")
    print(f"Parameter count: {model.count_parameters():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
