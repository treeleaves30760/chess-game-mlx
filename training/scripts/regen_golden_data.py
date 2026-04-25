"""
Regenerate golden data for encoding spec tests.

Generates ``shared/golden_data/chess_positions.json`` and
``shared/golden_data/chess_encodings.npz`` from a diverse set of chess
positions.  Run this whenever the encoding spec changes.

Usage::

    uv run python -m training.scripts.regen_golden_data
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np

from training.data.encoding import encode_chess_position

# ---------------------------------------------------------------------------
# Curated FEN strings covering various position types
# ---------------------------------------------------------------------------

_CHESS_FENS: list[str] = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # After 1.e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # After 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    # Sicilian after 1.e4 c5 2.Nf3
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    # Castled position
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # After kingside castle
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 b - - 2 6",
    # En passant possible
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    # Promotion possible (white pawn on 7th rank)
    "8/P7/8/8/8/8/8/4K2k w - - 0 1",
    # Endgame — kings and pawns
    "4k3/8/8/3p4/3P4/8/8/4K3 w - - 0 1",
    # Stalemate position (black to move, stalemated)
    "k7/8/1Q6/8/8/8/8/4K3 b - - 0 1",
    # Complex middlegame with all castling rights intact
    "r2qkb1r/ppp2ppp/2npbn2/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 7",
    # Position with no castling rights
    "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",
    # White queen-side castling right only
    "r3k2r/8/8/8/8/8/8/R3K2R w Q - 0 1",
    # Black to move with en passant
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
    # High half-move clock (50-move rule territory)
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 48 1",
    # Rook endgame
    "8/8/3k4/8/8/3K4/3R4/8 w - - 0 1",
    # Both sides have pieces on same file
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
    # Lone kings
    "8/8/3k4/8/8/3K4/8/8 w - - 0 1",
    # Many pieces on the board
    "rnbqkbnr/pppppppp/8/8/8/NNBBRRQQ/PPPPPPPP/R3K2R w KQ - 0 1",
    # Black bishop on long diagonal
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPPBPPP/RNBQK1NR w KQkq - 2 3",
]


def regen_chess_golden(output_dir: Path) -> None:
    """Generate chess golden data files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    positions_path = output_dir / "chess_positions.json"
    encodings_path = output_dir / "chess_encodings.npz"

    # Encode all positions
    encodings: list[np.ndarray] = []
    for fen in _CHESS_FENS:
        board = chess.Board(fen)
        enc = encode_chess_position(board)
        assert enc.shape == (64, 19), f"Wrong shape for {fen}: {enc.shape}"
        assert enc.dtype == np.float32
        encodings.append(enc)

    # Save JSON
    with open(positions_path, "w") as f:
        json.dump({"version": "v1.0", "positions": _CHESS_FENS}, f, indent=2)

    # Save NPZ
    arr = np.stack(encodings)  # [N, 64, 19]
    np.savez(str(encodings_path), encodings=arr, fens=np.array(_CHESS_FENS))

    print(f"Wrote {len(_CHESS_FENS)} chess positions to {positions_path}")
    print(f"Wrote encodings ({arr.shape}) to {encodings_path}")


def main() -> None:
    """Regenerate all golden data."""
    project_root = Path(__file__).resolve().parents[2]
    golden_dir = project_root / "shared" / "golden_data"

    print(f"Golden data directory: {golden_dir}")
    regen_chess_golden(golden_dir)
    print("Done.")


if __name__ == "__main__":
    main()
