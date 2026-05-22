"""Tests for AnalysisPanel mate handling.

Focus: the mate banner must stay visible across brief windows where the engine
flickers back to a `score cp` line (which happens because the C++ MCTS reporter
can read a node's running average mid-backprop and momentarily see |q| < 0.99).
It must, however, clear on reset() — i.e. when the position is known to change.
"""

from __future__ import annotations

import os

import pytest

# Force headless SDL before importing pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from gui.engine_client import InfoUpdate


@pytest.fixture(scope="module", autouse=True)
def pygame_init():
    pygame.init()
    yield
    pygame.quit()


def _make_panel():
    from gui.analysis_panel import AnalysisPanel

    surface = pygame.Surface((800, 600))
    return AnalysisPanel(
        surface=surface,
        right_panel_rect=pygame.Rect(600, 0, 200, 600),
        eval_bar_rect=pygame.Rect(0, 0, 24, 480),
    )


def test_mate_banner_persists_across_cp_flicker() -> None:
    panel = _make_panel()

    # Engine reports a forced mate.
    panel.update_info(InfoUpdate(depth=1, multipv=1, score_mate=3, pv=["a1a8"]))
    assert panel._detect_mate() == (3, True)

    # Next reporter cycle flickers back to a plain centipawn score (the bug:
    # a transient |q| < 0.99 read). Banner must NOT vanish.
    panel.update_info(InfoUpdate(depth=2, multipv=1, score_cp=1200, pv=["a1a8"]))
    assert panel._detect_mate() == (3, True), "mate banner should be sticky"

    # And it re-affirms on the following cycle.
    panel.update_info(InfoUpdate(depth=3, multipv=1, score_mate=2, pv=["a1a8"]))
    assert panel._detect_mate() == (2, True)

    panel.render()  # smoke: must not raise with sticky mate active


def test_mate_banner_clears_on_reset() -> None:
    panel = _make_panel()
    panel.update_info(InfoUpdate(depth=1, multipv=1, score_mate=5, pv=["h7h8"]))
    assert panel._detect_mate() == (5, True)

    panel.reset()  # position changed (move played / undo / new game)
    assert panel._detect_mate() is None

    # A subsequent ordinary eval is shown normally, no stale banner.
    panel.update_info(InfoUpdate(depth=4, multipv=1, score_cp=230, pv=["e2e4"]))
    assert panel._detect_mate() is None
    assert panel._score_cp == 230


def test_empty_pv_update_does_not_blank_top_move() -> None:
    """A PV-less update must not overwrite a real top-1 entry.

    Regression for "the #1 move vanishes, leaving only 2. and 3.": a bogus
    InfoUpdate(multipv=1, pv=[]) (e.g. from a mis-parsed `info string` line)
    used to clobber infos[0], so _draw_top_moves skipped slot 1.
    """
    panel = _make_panel()
    panel.update_info(InfoUpdate(depth=8, multipv=1, score_cp=120, pv=["e2e4"]))
    panel.update_info(InfoUpdate(depth=8, multipv=2, score_cp=90, pv=["d2d4"]))
    panel.update_info(InfoUpdate(depth=8, multipv=3, score_cp=50, pv=["c2c4"]))

    # A PV-less multipv-1 update arrives (the failure mode). It must be ignored.
    panel.update_info(InfoUpdate(depth=8, multipv=1, score_cp=0, pv=[]))

    assert panel._infos[0].pv == ["e2e4"], "top-1 move must survive a PV-less update"
    assert panel._score_cp == 120, "eval must not be yanked to 0 by a PV-less line"
    panel.render()  # smoke


def test_losing_mate_marked_not_winning() -> None:
    panel = _make_panel()
    panel.update_info(InfoUpdate(depth=1, multipv=1, score_mate=-4, pv=["g1f1"]))
    assert panel._detect_mate() == (4, False)


def test_saturated_cp_fallback_detected_as_mate() -> None:
    """Older engine builds emit saturated cp (>= 29000) instead of `score mate`."""
    panel = _make_panel()
    panel.update_info(InfoUpdate(depth=1, multipv=1, score_cp=31000, pv=["a1a8"]))
    mate = panel._detect_mate()
    assert mate is not None
    n, winning = mate
    assert winning is True
    assert n >= 1
