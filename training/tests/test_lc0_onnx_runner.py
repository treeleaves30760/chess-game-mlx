"""End-to-end tests for the LC0 ONNX runner and its 112-plane encoder.

These tests require a pre-exported ONNX network (produced by
``lc0 leela2onnx --input=<weights.pb.gz> --output=<file.onnx>``). Default
location is ``/tmp/lc0_onnx/t1_256.onnx``; override with env variable
``LC0_ONNX_T1_PATH``.

If the ONNX file isn't present, tests are skipped. The network itself is
not checked into the repo (it's 80+ MB).
"""

from __future__ import annotations

import os
from pathlib import Path

import chess
import numpy as np
import pytest

DEFAULT_ONNX = Path(os.environ.get("LC0_ONNX_T1_PATH", "/tmp/lc0_onnx/t1_256.onnx"))


@pytest.fixture(scope="module")
def runner():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("chess")
    if not DEFAULT_ONNX.exists():
        pytest.skip(
            f"ONNX network not found at {DEFAULT_ONNX}. "
            "Export with: lc0 leela2onnx --input=data/lc0_nets/t1_256_distilled.pb.gz "
            f"--output={DEFAULT_ONNX}"
        )
    from training.lc0.onnx_runner import LC0OnnxRunner

    return LC0OnnxRunner(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])


def test_startpos_top_move_is_mainline(runner):
    """On startpos, the top move should be d2d4, g1f3, or e2e4 (all common
    opening moves for a well-trained network)."""
    out = runner.forward_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    order = np.argsort(-out.legal_priors)
    top_move = out.legal_moves[order[0]].uci()
    assert top_move in {"d2d4", "g1f3", "e2e4"}, (
        f"Top move {top_move} is not a canonical opening move"
    )


def test_wdl_sums_to_one(runner):
    out = runner.forward_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert abs(out.wdl.sum() - 1.0) < 1e-5


def test_black_to_move_mirror_consistency(runner):
    """Black-to-move positions should produce mirrored UCI moves.

    After 1.Nf3, the best black reply should be a common response like e7e5,
    d7d5, or Nf6 (g8f6).
    """
    out = runner.forward_fen(
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"
    )
    order = np.argsort(-out.legal_priors)
    top = out.legal_moves[order[0]].uci()
    assert top in {"d7d5", "e7e5", "g8f6", "c7c5", "e7e6"}, (
        f"Top move {top} is not a reasonable response"
    )


def test_en_passant_encoding(runner):
    """FEN with en-passant available: the encoding should use the 'undo pawn
    push' history trick, producing output consistent with lc0's classical
    112-plane encoder. Without the fix, e7e5 gets < 50% (the natural reply
    to e4). With the fix, e7e5 gets ~54% on t1_256."""
    out = runner.forward_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    order = np.argsort(-out.legal_priors)[:3]
    moves = {out.legal_moves[i].uci() for i in order}
    assert "e7e5" in moves
    # After fix, e7e5 is ~54% of probability
    e7e5_idx = next(i for i, m in enumerate(out.legal_moves) if m.uci() == "e7e5")
    assert out.legal_priors[e7e5_idx] > 0.40, (
        f"e7e5 prior {out.legal_priors[e7e5_idx]:.3f} is too low — "
        "e.p. history reconstruction may be broken"
    )


def test_policy_output_size(runner):
    out = runner.forward_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert out.policy.shape == (1858,)


def test_encoder_bitboard_orientation():
    """Check the plane layout directly: for startpos, our pawns should sit
    on row 1 (rank 2) and their pawns on row 6 (rank 7)."""
    from training.lc0.encoding import encode_fen

    enc = encode_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Plane 0 = our pawns, should be on rank 2 (row 1 in [8,8] layout)
    assert enc.planes[0, 1].sum() == 8.0  # 8 pawns on rank 2
    assert enc.planes[0, 0].sum() == 0.0  # no pawns on rank 1
    # Plane 6 = their pawns, rank 7 (row 6)
    assert enc.planes[6, 6].sum() == 8.0
    # Plane 111 = all-ones helper
    assert enc.planes[111].sum() == 64.0
    # Plane 109 = rule50 scalar (startpos has halfmove_clock=0)
    assert enc.planes[109].sum() == 0.0


def test_move_index_roundtrip():
    """A representative set of moves should roundtrip through
    ``move_to_nn_index`` and ``nn_index_to_uci``."""
    from training.lc0.move_index import move_to_nn_index, nn_index_to_uci

    for uci, is_black in [
        ("e2e4", False),
        ("g1f3", False),
        ("e7e5", True),
        ("a7a8q", False),  # white promotion
        ("a2a1q", True),   # black promotion (mirrors to a7a8q)
    ]:
        move = chess.Move.from_uci(uci)
        idx = move_to_nn_index(move, is_black)
        assert idx is not None, f"Move {uci} (black={is_black}) not in table"
        back = nn_index_to_uci(idx, is_black)
        assert back == uci, f"Roundtrip failed: {uci} → {idx} → {back}"


def test_policy_output_deterministic(runner):
    """Two calls with the same input should return identical outputs."""
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    a = runner.forward_fen(fen)
    b = runner.forward_fen(fen)
    assert np.allclose(a.policy, b.policy, atol=1e-6)
    assert np.allclose(a.wdl, b.wdl, atol=1e-6)
