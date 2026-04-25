"""Mapping between ``chess.Move`` and LC0's compact 1858-element policy index.

The ordering is copied verbatim from LC0's
``src/neural/encoder.cc`` (``kMoveStrs`` array, GPL v3).

Semantics from LC0's :func:`MoveToNNIndex`:

* All moves are expressed from the perspective of the side to move **as if
  they were White** — i.e. the NN always sees the position with "us" on
  ranks 1..2. For a black-to-move position, LC0 mirrors (flips ranks 1↔8
  and swaps the "us"/"them" color meaning). The policy output uses the same
  mirrored coordinate frame, so when you want to map a real ``chess.Move``
  (on the actual, possibly black-to-move board) into the NN index space,
  you must mirror the move if black is to move.

* Promotion moves are listed last (24 × 8 = 192 slots, actually 192
  promotion entries — q/r/b, not knight, because knight-promotion is the
  default and therefore "not a promotion" in LC0's packing). This means
  moves 0..1665 are non-promotion slides and moves 1666..1857 are
  promotions.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Optional

import chess

# -----------------------------------------------------------------------------
# Load the 1858 move strings from a data file shipped alongside this module.
# -----------------------------------------------------------------------------


def _load_move_strs() -> list[str]:
    path = resources.files("training.lc0") / "lc0_move_strs.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


@lru_cache(maxsize=1)
def _move_str_to_idx() -> dict[str, int]:
    strs = _load_move_strs()
    assert len(strs) == 1858, f"Expected 1858 moves, got {len(strs)}"
    return {s: i for i, s in enumerate(strs)}


@lru_cache(maxsize=1)
def _idx_to_move_str() -> list[str]:
    return _load_move_strs()


# -----------------------------------------------------------------------------
# Move mirroring (rank flip only — LC0 always mirrors by color-swap too, but
# for pure move *strings* what matters is the rank flip: a2a4 ↔ a7a5, and
# promotions on rank 7 ↔ promotions on rank 2).
# -----------------------------------------------------------------------------


def _mirror_square(sq: int) -> int:
    return sq ^ 56  # flip rank


def _mirror_move(move: chess.Move) -> chess.Move:
    """Flip ranks for both from/to squares. Promotion piece is unchanged."""
    return chess.Move(
        from_square=_mirror_square(move.from_square),
        to_square=_mirror_square(move.to_square),
        promotion=move.promotion,
    )


def _move_to_lc0_string(move: chess.Move) -> str:
    """Convert a ``chess.Move`` (already oriented from White's POV) to the
    string format used by LC0's ``kMoveStrs``: ``e2e4``, ``a7a8q``.

    LC0 only represents under-promotions (q, r, b) — knight promotions are
    **not** present in the 1858-move table because LC0 treats a knight
    promotion as "the default move with no promotion annotation".
    """
    base = chess.square_name(move.from_square) + chess.square_name(move.to_square)
    if move.promotion is None or move.promotion == chess.KNIGHT:
        return base
    # Promotion piece → suffix
    suffix_map = {chess.QUEEN: "q", chess.ROOK: "r", chess.BISHOP: "b"}
    suffix = suffix_map.get(move.promotion)
    if suffix is None:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}")
    return base + suffix


def move_to_nn_index(move: chess.Move, is_black_to_move: bool) -> Optional[int]:
    """Convert a ``chess.Move`` to its LC0 NN index.

    Args:
        move: The move on the real board (i.e. in the ``chess.Board``'s
            own coordinate frame).
        is_black_to_move: True if the side to move is black on the real
            board. Used to decide whether to mirror before lookup.

    Returns:
        The integer index in ``[0, 1857]``, or ``None`` if the move is not
        in the LC0 table (should not happen for legal chess moves).
    """
    if is_black_to_move:
        move = _mirror_move(move)
    s = _move_to_lc0_string(move)
    idx = _move_str_to_idx().get(s)
    if idx is not None:
        return idx
    # Try again assuming it's a knight promotion (LC0 strips ``n`` so the
    # move string would have no suffix). Only relevant if the caller built
    # the move with ``promotion=chess.KNIGHT``.
    if move.promotion == chess.KNIGHT:
        base = chess.square_name(move.from_square) + chess.square_name(move.to_square)
        return _move_str_to_idx().get(base)
    return None


def nn_index_to_uci(idx: int, is_black_to_move: bool) -> str:
    """Convert an LC0 NN index to a UCI move string on the *real* board.

    For a black-to-move position, the index refers to a White-oriented move;
    we mirror it back so callers can compare to python-chess output.
    """
    if not 0 <= idx < 1858:
        raise IndexError(f"NN index out of range: {idx}")
    s = _idx_to_move_str()[idx]
    from_sq = chess.parse_square(s[0:2])
    to_sq = chess.parse_square(s[2:4])
    promotion = None
    if len(s) == 5:
        promotion = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP}[s[4]]
    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if is_black_to_move:
        move = _mirror_move(move)
    return move.uci()


__all__ = ["move_to_nn_index", "nn_index_to_uci"]
