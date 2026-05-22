"""GameController stale-info barrier tests.

Regression: after undo (or any position change) the engine's prior search
thread typically emits one last `info ...` plus a `bestmove ...` before the
new `position` command takes effect. Without a barrier those late InfoUpdates
land in the GUI's event queue and repopulate the analysis panel with numbers
for the position the user just left, so undo "didn't change the analysis".

The barrier here pairs every `isready` we send with the engine's `readyok`
reply and drops any InfoUpdate received in between, since by construction it
was generated under the prior position.
"""

from __future__ import annotations

import chess

from gui.engine_client import (
    BestMoveEvent,
    InfoUpdate,
    ReadyOkEvent,
    UciOkEvent,
)
from gui.game_controller import GameController, GameMode


class _FakeClient:
    """Records the UCI lines the controller sends. No subprocess."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send_uci(self, line: str) -> None:
        self.sent.append(line)


def _new_controller(mode: GameMode = GameMode.ANALYSIS) -> tuple[GameController, _FakeClient]:
    client = _FakeClient()
    ctrl = GameController(client=client, game="chess", mode=mode)  # type: ignore[arg-type]
    return ctrl, client


def test_isready_barrier_drops_stale_info_after_position_change() -> None:
    ctrl, client = _new_controller()
    ctrl.initialize_engine()  # sends one isready → barrier = 1
    # Engine's startup readyok lands → barrier back to 0.
    assert ctrl.process_event(ReadyOkEvent()) is True
    assert ctrl._isready_outstanding == 0

    # A fresh InfoUpdate while barrier is clear should be stored.
    fresh = InfoUpdate(depth=5, multipv=1, score_cp=20, pv=["e2e4"])
    assert ctrl.process_event(fresh) is True
    assert ctrl._latest_info[1] is fresh

    # Simulate a human move — try_move sends position + isready, and in
    # ANALYSIS mode _start_analysis re-sends position too, so 2 barriers.
    assert ctrl.try_move(chess.Move.from_uci("e2e4"))
    assert ctrl._isready_outstanding == 2

    # try_move clears _latest_info before sending position, so the dict is
    # empty here; the barrier's job is to keep it empty (not let a stale
    # info repopulate it) until the engine acknowledges the new position.
    assert ctrl._latest_info == {}
    stale = InfoUpdate(depth=99, multipv=1, score_cp=-999, pv=["a7a6"])
    assert ctrl.process_event(stale) is False, "stale info must be dropped"
    assert ctrl._latest_info == {}, "stale info must not repopulate _latest_info"

    # Each readyok decrements the barrier; until it hits 0 infos stay stale.
    ctrl.process_event(ReadyOkEvent())
    assert ctrl._isready_outstanding == 1
    still_stale = InfoUpdate(depth=100, multipv=1, score_cp=-888, pv=["b7b6"])
    assert ctrl.process_event(still_stale) is False

    ctrl.process_event(ReadyOkEvent())
    assert ctrl._isready_outstanding == 0
    new_pos_info = InfoUpdate(depth=3, multipv=1, score_cp=15, pv=["e7e5"])
    assert ctrl.process_event(new_pos_info) is True
    assert ctrl._latest_info[1] is new_pos_info


def test_undo_drops_stale_info_from_interrupted_search() -> None:
    """The originally reported bug: undo + lingering info = stale panel."""
    ctrl, _ = _new_controller()
    ctrl.initialize_engine()
    ctrl.process_event(ReadyOkEvent())  # clear init barrier

    # Play a move and let the analysis search populate top-3.
    ctrl.try_move(chess.Move.from_uci("e2e4"))
    # Drain the two readyoks the move emitted (position + _start_analysis).
    ctrl.process_event(ReadyOkEvent())
    ctrl.process_event(ReadyOkEvent())
    assert ctrl._isready_outstanding == 0

    populated = InfoUpdate(depth=20, multipv=1, score_cp=42, pv=["d7d5"])
    assert ctrl.process_event(populated)
    assert ctrl._latest_info[1] is populated

    # User undoes.
    assert ctrl.undo()
    # Barrier should be armed (undo's _send_position + _start_analysis's _send_position).
    assert ctrl._isready_outstanding >= 1

    # Engine flushes its last "info" + "bestmove" for the interrupted search.
    flush_info = InfoUpdate(depth=21, multipv=1, score_cp=999, pv=["d7d5", "e2e4"])
    accepted = ctrl.process_event(flush_info)
    assert accepted is False, "post-undo flush info must be dropped"
    # The pre-undo info must NOT be overwritten by the flush — it'll be
    # cleared explicitly by undo() via _latest_info.clear(), but the panel
    # state in game_scene relies on the False return to skip its own update.
    # _latest_info was cleared by undo(); confirm it stayed empty.
    assert ctrl._latest_info == {}

    # Bestmove from the interrupted search is not an InfoUpdate, so the
    # barrier doesn't apply — handled by mode-specific bestmove logic.
    ctrl.process_event(BestMoveEvent(move="d7d5"))  # should not blow up

    # After the matching readyoks arrive, the next info is honoured.
    while ctrl._isready_outstanding > 0:
        ctrl.process_event(ReadyOkEvent())
    new_info = InfoUpdate(depth=2, multipv=1, score_cp=10, pv=["e7e5"])
    assert ctrl.process_event(new_info)
    assert ctrl._latest_info[1] is new_info


def test_uciok_and_bestmove_not_affected_by_barrier() -> None:
    """Non-InfoUpdate events must always be processed."""
    ctrl, _ = _new_controller()
    ctrl._isready_outstanding = 3  # simulate a deeply-armed barrier
    # UciOkEvent should still set engine name etc.
    assert ctrl.process_event(UciOkEvent(engine_name="TestEngine")) is True
    assert ctrl.state.engine_name == "TestEngine"
    # BestMoveEvent processing doesn't depend on the barrier.
    ctrl.process_event(BestMoveEvent(move="e2e4"))
    # Barrier untouched by either.
    assert ctrl._isready_outstanding == 3
