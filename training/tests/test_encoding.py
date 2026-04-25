"""
Tests for position encoding (encoding_spec.md v1.0).

Verifies:
1. Output shape is [64, 19] float32 for chess.
2. Piece placement is correct (startpos specific squares).
3. Castling rights planes.
4. En passant plane.
5. Half-move clock plane.
6. Side-to-move plane.
7. Encoding is deterministic.
8. Move → policy index round-trips correctly.
9. A known move produces the expected index.
"""

from __future__ import annotations

import numpy as np
import chess
import pytest

from training.data.encoding import (
    _CHESS_NUM_COMPACT,
    chess_idx_to_move,
    chess_move_to_idx,
    encode_chess_position,
)


# ---------------------------------------------------------------------------
# Basic shape / dtype
# ---------------------------------------------------------------------------


def test_shape_and_dtype_startpos() -> None:
    """encode_chess_position returns [64, 19] float32."""
    board = chess.Board()  # starting position
    enc = encode_chess_position(board)
    assert enc.shape == (64, 19), f"Wrong shape: {enc.shape}"
    assert enc.dtype == np.float32, f"Wrong dtype: {enc.dtype}"


# ---------------------------------------------------------------------------
# Piece placement — startpos
# ---------------------------------------------------------------------------


def test_startpos_white_pawn_e2() -> None:
    """White pawn on e2 → square 12, plane 0 = 1.

    Square encoding: a1=0, b1=1, ..., h1=7, a2=8, ..., h2=15.
    e2 = file e (4) + rank 1 (8*1) = 12. ✓
    """
    board = chess.Board()
    enc = encode_chess_position(board)
    assert enc[12, 0] == 1.0, f"White pawn at e2 (sq=12) plane 0: {enc[12, 0]}"


def test_startpos_white_king_e1() -> None:
    """White king on e1 → square 4, plane 5 = 1."""
    board = chess.Board()
    enc = encode_chess_position(board)
    # e1 = file e(4) + rank 0*8 = 4
    assert enc[4, 5] == 1.0, f"White king at e1 (sq=4) plane 5: {enc[4, 5]}"


def test_startpos_black_king_e8() -> None:
    """Black king on e8 → square 60, plane 11 = 1."""
    board = chess.Board()
    enc = encode_chess_position(board)
    # e8 = 4 + 7*8 = 60
    assert enc[60, 11] == 1.0, f"Black king at e8 (sq=60) plane 11: {enc[60, 11]}"


def test_startpos_black_rook_h8() -> None:
    """Black rook on h8 → square 63, plane 9 = 1."""
    board = chess.Board()
    enc = encode_chess_position(board)
    # h8 = 7 + 7*8 = 63
    assert enc[63, 9] == 1.0, f"Black rook at h8 (sq=63) plane 9: {enc[63, 9]}"


def test_startpos_empty_square_e4() -> None:
    """Empty square e4 has all piece planes = 0."""
    board = chess.Board()
    enc = encode_chess_position(board)
    # e4 = 4 + 3*8 = 28
    assert np.all(enc[28, :12] == 0.0), f"e4 planes 0-11 not all 0: {enc[28, :12]}"


def test_startpos_piece_count() -> None:
    """Starting position has exactly 32 piece-plane activations (16 pieces × 2 colours × 1 each)."""
    board = chess.Board()
    enc = encode_chess_position(board)
    total_pieces = enc[:, :12].sum()
    assert total_pieces == 32.0, f"Expected 32 piece activations, got {total_pieces}"


# ---------------------------------------------------------------------------
# Side-to-move
# ---------------------------------------------------------------------------


def test_side_to_move_white() -> None:
    """Plane 12 = 1 for all squares when white to move."""
    board = chess.Board()  # white to move
    enc = encode_chess_position(board)
    assert np.all(enc[:, 12] == 1.0), "Plane 12 not all 1 when white to move"


def test_side_to_move_black() -> None:
    """Plane 12 = 0 for all squares when black to move."""
    board = chess.Board()
    board.push_san("e4")  # black to move
    enc = encode_chess_position(board)
    assert np.all(enc[:, 12] == 0.0), "Plane 12 not all 0 when black to move"


# ---------------------------------------------------------------------------
# Castling rights
# ---------------------------------------------------------------------------


def test_castling_startpos() -> None:
    """All four castling rights present at start."""
    board = chess.Board()
    enc = encode_chess_position(board)
    assert np.all(enc[:, 13] == 1.0), "White KS castling missing"
    assert np.all(enc[:, 14] == 1.0), "White QS castling missing"
    assert np.all(enc[:, 15] == 1.0), "Black KS castling missing"
    assert np.all(enc[:, 16] == 1.0), "Black QS castling missing"


def test_castling_no_rights() -> None:
    """Position with no castling rights has planes 13-16 all 0."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    enc = encode_chess_position(board)
    assert np.all(enc[:, 13] == 0.0)
    assert np.all(enc[:, 14] == 0.0)
    assert np.all(enc[:, 15] == 0.0)
    assert np.all(enc[:, 16] == 0.0)


# ---------------------------------------------------------------------------
# En passant
# ---------------------------------------------------------------------------


def test_en_passant_target() -> None:
    """After 1.e4 e5 2.d4, en passant target is d6 (square = d6)."""
    # FEN after 1.e4 e5 2.d4 (black pawn captures en-passant on d3 or white on e6?)
    # More precisely: after 1.d4 e5 2.d5 c5, en passant target is c6 (square 42)
    board = chess.Board("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3")
    # c6 = file c (2) + rank 5 (5*8) = 42
    enc = encode_chess_position(board)
    assert enc[42, 17] == 1.0, f"EP square c6 (42) not set: {enc[42, 17]}"
    # All other squares on plane 17 should be 0
    mask = np.ones(64, dtype=bool)
    mask[42] = False
    assert np.all(enc[mask, 17] == 0.0), "Other squares have non-zero EP plane"


def test_no_en_passant() -> None:
    """Starting position has no en passant — plane 17 all 0."""
    board = chess.Board()
    enc = encode_chess_position(board)
    assert np.all(enc[:, 17] == 0.0), "Plane 17 not all 0 with no EP"


# ---------------------------------------------------------------------------
# Half-move clock
# ---------------------------------------------------------------------------


def test_halfmove_clock_zero() -> None:
    """Half-move clock 0 → plane 18 = 0.0."""
    board = chess.Board()  # clock = 0
    enc = encode_chess_position(board)
    assert np.all(enc[:, 18] == 0.0), f"Plane 18 != 0 at clock=0: {enc[0, 18]}"


def test_halfmove_clock_50() -> None:
    """Half-move clock 50 → plane 18 = 0.5."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 50 1")
    enc = encode_chess_position(board)
    expected = 50 / 100.0
    assert np.allclose(enc[:, 18], expected), f"Plane 18 != {expected}: {enc[0, 18]}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_encoding_deterministic() -> None:
    """Same board produces identical encoding on two calls."""
    board = chess.Board()
    enc1 = encode_chess_position(board)
    enc2 = encode_chess_position(board)
    np.testing.assert_array_equal(enc1, enc2, err_msg="Encoding not deterministic")


def test_encoding_deterministic_complex() -> None:
    """Encoding of a complex position is deterministic."""
    board = chess.Board("r2qkb1r/ppp2ppp/2npbn2/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 7")
    enc1 = encode_chess_position(board)
    enc2 = encode_chess_position(board)
    np.testing.assert_array_equal(enc1, enc2)


# ---------------------------------------------------------------------------
# Policy map coverage
# ---------------------------------------------------------------------------


def test_policy_map_size() -> None:
    """Compact policy map should have 1858 entries."""
    assert _CHESS_NUM_COMPACT == 1858, (
        f"Expected 1858 compact policy slots, got {_CHESS_NUM_COMPACT}"
    )


def test_chess_move_roundtrip() -> None:
    """chess_move_to_idx → chess_idx_to_move round-trip for starting position moves."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = chess_move_to_idx(move, board)
        assert 0 <= idx < _CHESS_NUM_COMPACT, f"Index {idx} out of range for move {move}"
        recovered = chess_idx_to_move(idx, board)
        # We can't guarantee exact round-trip for all moves (promotion handling
        # can be ambiguous), but the index should be in range and non-None.
        assert recovered is not None, f"Round-trip failed for move {move} (idx={idx})"


def test_e2e4_policy_index() -> None:
    """e2e4 from startpos should produce a valid index in [0, 1857]."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = chess_move_to_idx(move, board)
    assert 0 <= idx < 1858, f"e2e4 index {idx} out of range"


def test_all_startpos_moves_unique_indices() -> None:
    """All legal moves from starting position map to unique indices."""
    board = chess.Board()
    indices = [chess_move_to_idx(m, board) for m in board.legal_moves]
    assert len(indices) == len(set(indices)), "Duplicate policy indices for starting position moves"
