"""Tests for EngineClient + MockEngine integration."""

from __future__ import annotations

import time

import pytest
from gui.engine_client import (
    BestMoveEvent,
    EngineClient,
    EngineEvent,
    InfoUpdate,
    ReadyOkEvent,
    UciOkEvent,
)


def _drain_until(
    client: EngineClient,
    pred: type,
    timeout: float = 5.0,
) -> EngineEvent | None:
    """Poll until an event of type *pred* appears or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in client.poll_events():
            if isinstance(ev, pred):
                return ev
        time.sleep(0.05)
    return None


@pytest.fixture()
def mock_client() -> EngineClient:
    """Start a mock engine client and stop it after the test."""
    client = EngineClient("mock", "chess")
    client.start()
    yield client
    client.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_uciok_received(mock_client: EngineClient) -> None:
    """Send 'uci' and expect uciok within 2 seconds."""
    mock_client.send_uci("uci")
    ev = _drain_until(mock_client, UciOkEvent, timeout=2.0)
    assert ev is not None, "uciok not received within 2 seconds"
    assert isinstance(ev, UciOkEvent)


def test_readyok_received(mock_client: EngineClient) -> None:
    """Send 'isready' and expect readyok."""
    mock_client.send_uci("uci")
    mock_client.send_uci("isready")
    ev = _drain_until(mock_client, ReadyOkEvent, timeout=2.0)
    assert ev is not None, "readyok not received within 2 seconds"


def test_go_emits_info_and_bestmove(mock_client: EngineClient) -> None:
    """Send position + go; expect at least one InfoUpdate and one BestMoveEvent."""
    mock_client.send_uci("uci")
    _drain_until(mock_client, UciOkEvent, timeout=2.0)
    mock_client.send_uci("isready")
    _drain_until(mock_client, ReadyOkEvent, timeout=2.0)

    mock_client.send_uci("position startpos")
    mock_client.send_uci("go movetime 800")

    info_ev = _drain_until(mock_client, InfoUpdate, timeout=3.0)
    assert info_ev is not None, "No InfoUpdate received"
    assert isinstance(info_ev, InfoUpdate)
    assert info_ev.depth >= 1

    best_ev = _drain_until(mock_client, BestMoveEvent, timeout=5.0)
    assert best_ev is not None, "No BestMoveEvent received"
    assert isinstance(best_ev, BestMoveEvent)
    assert best_ev.move  # non-empty


def test_parse_info_line() -> None:
    """Unit test for the UCI info line parser."""
    from gui.engine_client import parse_info_line

    line = (
        "info depth 12 seldepth 14 multipv 2 score cp -35 "
        "nodes 120000 nps 80000 time 1500 pv e2e4 e7e5 g1f3"
    )
    info = parse_info_line(line)
    assert info is not None
    assert info.depth == 12
    assert info.seldepth == 14
    assert info.multipv == 2
    assert info.score_cp == -35
    assert info.nodes == 120_000
    assert info.nps == 80_000
    assert info.time_ms == 1_500
    assert info.pv == ["e2e4", "e7e5", "g1f3"]


def test_parse_info_line_mate() -> None:
    """Parse a mate score."""
    from gui.engine_client import parse_info_line

    line = "info depth 5 score mate 3 pv e2e4"
    info = parse_info_line(line)
    assert info is not None
    assert info.score_mate == 3
    assert info.pv == ["e2e4"]


def test_parse_info_string_is_ignored() -> None:
    """`info string ...` is free-form engine text, not a search line.

    Regression: the engine emits `info string batcher ...` right before every
    bestmove. Parsing it as a search line produced InfoUpdate(multipv=1, pv=[],
    score_cp=0), which overwrote the real top-1 move in the analysis panel (so
    the #1 move vanished, leaving only "2." and "3.") and zeroed the eval.
    """
    from gui.engine_client import parse_info_line

    assert parse_info_line("info string batcher batches=20 requests=289 avg_batch=14.4") is None
    assert parse_info_line(
        "info string setoption NN_Weights = foo (requires engine restart)"
    ) is None
    # A genuine search line that merely *contains* the word "string" later on is
    # unaffected (only the immediate "info string" prefix is special).
    info = parse_info_line("info depth 4 multipv 1 score cp 12 pv e2e4")
    assert info is not None
    assert info.pv == ["e2e4"]


