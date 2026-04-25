"""Smoke tests for ChessBoardRenderer.

We render to an in-memory Surface and assert no exceptions are raised.
No pixel-level assertions — just verifying the rendering pipeline works.
"""

from __future__ import annotations

import os

import pytest

# Force headless SDL before importing pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import chess
import pygame


@pytest.fixture(scope="module", autouse=True)
def pygame_init():
    """Initialize and quit pygame for the entire module."""
    pygame.init()
    yield
    pygame.quit()


def _make_surface(w: int = 400, h: int = 400) -> pygame.Surface:
    return pygame.Surface((w, h))


def test_startpos_render_no_crash() -> None:
    """Render the starting position without errors."""
    from gui.chess_board import ChessBoardRenderer

    surface = _make_surface()
    rect = pygame.Rect(0, 0, 400, 400)
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=chess.Board())
    renderer.render()  # should not raise


def test_render_specific_position() -> None:
    """Render an arbitrary FEN position."""
    from gui.chess_board import ChessBoardRenderer

    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    surface = _make_surface()
    rect = pygame.Rect(0, 0, 400, 400)
    board = chess.Board(fen)
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=board)
    renderer.render()


def test_render_with_selection_and_legal_moves() -> None:
    """Select a piece and render legal move dots."""
    from gui.chess_board import ChessBoardRenderer

    surface = _make_surface()
    rect = pygame.Rect(0, 0, 400, 400)
    board = chess.Board()
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=board)

    # Select e2 pawn
    renderer.set_selected(chess.E2)
    renderer.render()


def test_render_with_arrows() -> None:
    """Render with top-3 move arrows."""
    from gui.chess_board import ChessBoardRenderer, MoveArrow

    surface = _make_surface()
    rect = pygame.Rect(0, 0, 400, 400)
    board = chess.Board()
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=board)

    arrows = [
        MoveArrow(from_sq=chess.E2, to_sq=chess.E4, rank=1, score_cp=30),
        MoveArrow(from_sq=chess.D2, to_sq=chess.D4, rank=2, score_cp=20),
        MoveArrow(from_sq=chess.G1, to_sq=chess.F3, rank=3, score_cp=15),
    ]
    renderer.set_arrows(arrows)
    renderer.render()


def test_render_flipped() -> None:
    """Render with board flipped (black on bottom)."""
    from gui.chess_board import ChessBoardRenderer

    surface = _make_surface()
    rect = pygame.Rect(0, 0, 400, 400)
    renderer = ChessBoardRenderer(
        surface=surface, rect=rect, board=chess.Board(), flipped=True
    )
    renderer.render()


def test_square_at_pixel() -> None:
    """Test square detection from pixel coordinates."""
    from gui.chess_board import ChessBoardRenderer

    surface = _make_surface(400, 400)
    rect = pygame.Rect(0, 0, 400, 400)
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=chess.Board())

    # Top-left should be a8 (rank 7, file 0)
    sq = renderer.square_at_pixel(5, 5)
    assert sq is not None
    assert chess.square_file(sq) == 0
    assert chess.square_rank(sq) == 7

    # Bottom-right should be h1
    sq2 = renderer.square_at_pixel(395, 395)
    assert sq2 is not None
    assert chess.square_file(sq2) == 7
    assert chess.square_rank(sq2) == 0


def test_pixel_center_of_square() -> None:
    """Pixel centers should be inside the board rect."""
    from gui.chess_board import ChessBoardRenderer

    surface = _make_surface(400, 400)
    rect = pygame.Rect(0, 0, 400, 400)
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=chess.Board())

    for sq in chess.SQUARES:
        cx, cy = renderer.pixel_center_of_square(sq)
        assert rect.left <= cx < rect.right, f"cx {cx} out of board for {chess.square_name(sq)}"
        assert rect.top <= cy < rect.bottom, f"cy {cy} out of board for {chess.square_name(sq)}"


def test_render_check_position() -> None:
    """Render a position with the king in check."""
    from gui.chess_board import ChessBoardRenderer

    # Scholar's mate position (white in check)
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    surface = _make_surface(400, 400)
    rect = pygame.Rect(0, 0, 400, 400)
    board = chess.Board(fen)
    assert board.is_check()
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=board)
    renderer.render()


def test_render_after_moves() -> None:
    """Play several moves and render each position."""
    from gui.chess_board import ChessBoardRenderer

    surface = _make_surface()
    rect = pygame.Rect(0, 0, 400, 400)
    board = chess.Board()
    renderer = ChessBoardRenderer(surface=surface, rect=rect, board=board)

    moves_uci = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        board.push(move)
        renderer.set_board(board)
        renderer.set_last_move(move)
        renderer.render()


def test_shogi_board_render_no_crash() -> None:
    """Shogi stub renderer should not crash."""
    from gui.shogi_board import ShogiBoardRenderer

    surface = _make_surface(500, 500)
    rect = pygame.Rect(0, 0, 500, 500)
    renderer = ShogiBoardRenderer(surface=surface, rect=rect)
    renderer.render()
