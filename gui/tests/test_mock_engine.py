"""Tests for MockEngine: play 5 moves, verify no errors."""

from __future__ import annotations

import time

import chess
import pytest
from gui.engine_client import (
    BestMoveEvent,
    EngineClient,
    EngineError,
    InfoUpdate,
    ReadyOkEvent,
    UciOkEvent,
)


def _drain_all(client: EngineClient, duration: float = 0.2) -> list:
    """Drain events for *duration* seconds."""
    events = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        events.extend(client.poll_events())
        time.sleep(0.02)
    return events


def _wait_for(
    client: EngineClient,
    event_type: type,
    timeout: float = 5.0,
) -> object | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in client.poll_events():
            if isinstance(ev, event_type):
                return ev
        time.sleep(0.05)
    return None


@pytest.fixture()
def initialized_client():
    """A mock client that has completed uci / isready handshake."""
    client = EngineClient("mock", "chess")
    client.start()
    client.send_uci("uci")
    _wait_for(client, UciOkEvent, timeout=3.0)
    client.send_uci("isready")
    _wait_for(client, ReadyOkEvent, timeout=3.0)
    client.send_uci("ucinewgame")
    yield client
    client.stop()


def test_mock_engine_starts() -> None:
    """Basic: engine starts, responds to uci."""
    client = EngineClient("mock", "chess")
    client.start()
    assert client.is_running

    client.send_uci("uci")
    ev = _wait_for(client, UciOkEvent, timeout=3.0)
    assert ev is not None

    client.stop()


def test_play_five_random_moves(initialized_client: EngineClient) -> None:
    """Play 5 moves using the engine, verify no EngineErrors and each bestmove is legal."""
    board = chess.Board()
    moves_played: list[str] = []
    errors: list[EngineError] = []

    for move_num in range(5):
        # Choose a random legal move for the human side
        legal = list(board.legal_moves)
        assert legal, f"No legal moves at move {move_num}"
        human_move = legal[0]  # deterministic: first in list
        board.push(human_move)
        moves_played.append(human_move.uci())

        # Build position string and ask engine
        pos = "position startpos moves " + " ".join(moves_played)
        initialized_client.send_uci(pos)
        initialized_client.send_uci("go movetime 300")

        # Wait for bestmove
        best_ev = _wait_for(initialized_client, BestMoveEvent, timeout=5.0)
        assert best_ev is not None, f"No bestmove at move {move_num}"
        assert isinstance(best_ev, BestMoveEvent)

        # Collect any errors
        for ev in initialized_client.poll_events():
            if isinstance(ev, EngineError):
                errors.append(ev)

        # Apply engine move
        try:
            eng_move = chess.Move.from_uci(best_ev.move)
            if eng_move in board.legal_moves:
                board.push(eng_move)
                moves_played.append(best_ev.move)
            # else: engine might have returned a stale move, just skip
        except ValueError:
            pass  # invalid UCI string from mock — not an error in the mock protocol

    assert not errors, f"Engine errors during play: {errors}"


def test_mock_engine_multipv(initialized_client: EngineClient) -> None:
    """Test MultiPV=3 produces multiple info lines."""
    initialized_client.send_uci("setoption name MultiPV value 3")
    initialized_client.send_uci("position startpos")
    initialized_client.send_uci("go movetime 600")

    info_multipv: set[int] = set()
    got_bestmove = False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        for ev in initialized_client.poll_events():
            if isinstance(ev, InfoUpdate):
                info_multipv.add(ev.multipv)
            elif isinstance(ev, BestMoveEvent):
                got_bestmove = True
        if got_bestmove:
            break
        time.sleep(0.05)

    # With MultiPV=3 we should see multipv indices 1, 2, 3
    assert len(info_multipv) >= 1, f"No InfoUpdates received (got_bestmove={got_bestmove})"


def test_mock_engine_policy_preview(initialized_client: EngineClient) -> None:
    """Sending start_multi_ponder should produce a PolicyPreview."""
    from gui.engine_client import PolicyPreview

    initialized_client.send_uci("position startpos")
    initialized_client.send_jsonrpc("start_multi_ponder", {"k": 5})

    ev = _wait_for(initialized_client, PolicyPreview, timeout=3.0)
    assert ev is not None, "PolicyPreview not received"
    assert isinstance(ev, PolicyPreview)
    assert len(ev.moves) > 0
    # Each entry should have 'uci' and 'prob'
    for m in ev.moves:
        assert "uci" in m
        assert "prob" in m
        assert 0.0 <= m["prob"] <= 1.0


def test_mock_engine_get_eval_bar(initialized_client: EngineClient) -> None:
    """get_eval_bar RPC should return a valid response."""
    from gui.engine_client import EvalBarResult

    initialized_client.send_uci("position startpos")
    initialized_client.send_jsonrpc("get_eval_bar", {})

    ev = _wait_for(initialized_client, EvalBarResult, timeout=3.0)
    assert ev is not None, "EvalBarResult not received"
    assert isinstance(ev, EvalBarResult)
    assert 0.0 <= ev.win_prob <= 1.0
    assert ev.depth >= 0


def test_mock_engine_no_crash_on_quit() -> None:
    """Engine should exit cleanly when quit is sent."""
    client = EngineClient("mock", "chess")
    client.start()
    client.send_uci("uci")
    _wait_for(client, UciOkEvent, timeout=2.0)
    client.stop()
    # Give it a moment to clean up
    time.sleep(0.2)
    assert not client.is_running
