"""Game controller — orchestrates state transitions for chess or shogi.

Manages:
- Game mode (analysis / single-side / AI vs AI / standard)
- Turn tracking (my turn / opponent turn / pondering)
- Interaction with the engine client (sending UCI/USI commands)
- Propagating engine events to UI state (board, analysis panel, ponder manager)

The controller treats the chosen game transparently: `self.board` is either a
`chess.Board` or a `shogi.Board`, and helper methods dispatch on `self._game`
where the protocols diverge (UCI vs USI, startpos vs sfen startpos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import chess
import shogi

from gui.engine_client import (
    BestMoveEvent,
    EngineClient,
    EngineError,
    EngineEvent,
    InfoUpdate,
    ReadyOkEvent,
    UciOkEvent,
)


class GameMode(Enum):
    ANALYSIS = auto()         # go infinite, always analysing
    SINGLE_SIDE = auto()      # engine plays only one color (Human vs AI)
    AI_VS_AI = auto()         # engine plays both colors, alternating turns
    STANDARD = auto()         # human vs human (engine only shows eval bar)


class TurnPhase(Enum):
    MY_TURN = auto()
    OPP_TURN = auto()
    PONDERING = auto()
    IDLE = auto()


GameKind = Literal["chess", "shogi"]


@dataclass
class ControllerState:
    """Snapshot of game controller state for rendering."""

    mode: GameMode = GameMode.ANALYSIS
    turn_phase: TurnPhase = TurnPhase.IDLE
    # In SINGLE_SIDE: which color the engine plays. Chess uses chess.WHITE/BLACK,
    # shogi uses shogi.BLACK/WHITE — callers must use the game's own constants.
    my_side: int = chess.WHITE
    status_text: str = "Ready"
    engine_name: str = "Engine"
    is_engine_ready: bool = False
    move_history: list[str] = field(default_factory=list)  # UCI/USI strings
    # Whether the infinite analysis search is currently running. Only meaningful
    # in ANALYSIS mode — toggled by the Analysis button. When False, previously
    # computed InfoUpdates remain visible in the panel until the user resumes.
    analysis_active: bool = True


def _first_player(game: GameKind) -> int:
    """The side that moves first in the game's own color convention."""
    return chess.WHITE if game == "chess" else shogi.BLACK


def _second_player(game: GameKind) -> int:
    return chess.BLACK if game == "chess" else shogi.WHITE


def _side_label(game: GameKind, side: int) -> str:
    if game == "chess":
        return "white" if side == chess.WHITE else "black"
    return "sente" if side == shogi.BLACK else "gote"


class GameController:
    """Coordinates the game flow between UI and engine for chess or shogi."""

    def __init__(
        self,
        client: EngineClient,
        game: GameKind = "chess",
        mode: GameMode = GameMode.ANALYSIS,
        my_side: int | None = None,
    ) -> None:
        self._client = client
        self._game: GameKind = game
        self.board: Any = self._new_board()
        side = my_side if my_side is not None else _first_player(game)
        self.state = ControllerState(mode=mode, my_side=side)
        self._multi_pv = 3
        self._latest_info: dict[int, InfoUpdate] = {}
        # Position-change barrier. Every `isready` we send bumps this counter;
        # every `readyok` we receive decrements it. InfoUpdates that arrive
        # while the counter > 0 belong to a search the engine started *before*
        # the most recent position change (e.g. the final info+bestmove from a
        # search the user just interrupted with undo / a new move) and would
        # otherwise repopulate the analysis panel with stale data for a
        # position that no longer exists. We drop them.
        self._isready_outstanding = 0

    # -----------------------------------------------------------------------
    # Board construction / move formatting
    # -----------------------------------------------------------------------

    def _new_board(self) -> Any:
        return chess.Board() if self._game == "chess" else shogi.Board()

    def _move_to_str(self, move: Any) -> str:
        if self._game == "chess":
            return move.uci()
        return move.usi()

    def _move_from_str(self, s: str) -> Any:
        if self._game == "chess":
            return chess.Move.from_uci(s)
        return shogi.Move.from_usi(s)

    def _is_legal(self, move: Any) -> bool:
        return move in self.board.legal_moves

    def _position_cmd(self) -> str:
        moves_str = " ".join(self.state.move_history)
        if self._game == "chess":
            return (
                f"position startpos moves {moves_str}"
                if moves_str else "position startpos"
            )
        # USI uses "position startpos moves ..." too (same syntax)
        return (
            f"position startpos moves {moves_str}"
            if moves_str else "position startpos"
        )

    # -----------------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------------

    def initialize_engine(self) -> None:
        """Send UCI/USI init sequence. Call once after engine is started."""
        protocol = "uci" if self._game == "chess" else "usi"
        self._client.send_uci(protocol)
        self._send_isready()

        if self.state.mode in (GameMode.ANALYSIS, GameMode.SINGLE_SIDE, GameMode.AI_VS_AI):
            self._client.send_uci(f"setoption name MultiPV value {self._multi_pv}")

    def _send_isready(self) -> None:
        """Send `isready` and arm the post-position-change barrier.

        Pairs with the matching `readyok` decrement in process_event so the
        engine's reply order acts as a sync point — every InfoUpdate emitted
        *before* this readyok was generated under the prior position and is
        discarded by the caller of process_event.
        """
        self._client.send_uci("isready")
        self._isready_outstanding += 1

    def new_game(self) -> None:
        """Reset board and engine state."""
        # Halt any in-flight search first so ucinewgame's tree reset doesn't
        # race with a running search thread on the engine side.
        self._client.send_uci("stop")

        self.board = self._new_board()
        self._latest_info.clear()
        self.state.move_history = []
        self.state.status_text = self._initial_status()
        self.state.turn_phase = TurnPhase.IDLE

        self._client.send_uci(self._newgame_cmd())
        self._send_position()

        if self.state.mode == GameMode.ANALYSIS and self.state.analysis_active:
            self._start_analysis()
        elif self.state.mode == GameMode.AI_VS_AI:
            self.state.turn_phase = TurnPhase.MY_TURN
            self._kick_engine_search()
        elif self.state.mode == GameMode.SINGLE_SIDE:
            if self.board.turn == self.state.my_side:
                # Must set MY_TURN before kicking, otherwise the bestmove
                # arrives while turn_phase is IDLE and gets dropped by the
                # SINGLE_SIDE guard in _handle_bestmove — which leaves the
                # engine apparently "stuck searching" with no recovery.
                self.state.turn_phase = TurnPhase.MY_TURN
                self._kick_engine_search()
            else:
                self.state.turn_phase = TurnPhase.OPP_TURN

    def _initial_status(self) -> str:
        first = "White" if self._game == "chess" else "Sente (先手)"
        return f"New game — {first} to move"

    def _newgame_cmd(self) -> str:
        return "ucinewgame" if self._game == "chess" else "usinewgame"

    def _send_position(self) -> None:
        self._client.send_uci(self._position_cmd())
        # Arm the stale-info barrier — the engine may still have a search
        # thread finishing up from before this position change, and its
        # final info / bestmove must not be allowed to repopulate the UI
        # state we just rebuilt for the new position.
        self._send_isready()

    # -----------------------------------------------------------------------
    # Move input
    # -----------------------------------------------------------------------

    def try_move(self, move: Any) -> bool:
        """Attempt to make a move. Returns True if legal and applied."""
        if not self._is_legal(move):
            return False

        self.board.push(move)
        self.state.move_history.append(self._move_to_str(move))
        # Position changed — old MultiPV entries are stale, drop them so the
        # analysis panel doesn't show pre-move top-3 until fresh info arrives.
        self._latest_info.clear()
        self._send_position()

        # Post-move logic based on mode
        if self.state.mode == GameMode.ANALYSIS:
            if self.state.analysis_active:
                self._start_analysis()

        elif self.state.mode == GameMode.SINGLE_SIDE:
            if self.board.turn == self.state.my_side:
                self.state.turn_phase = TurnPhase.MY_TURN
                self._kick_engine_search()
                self.state.status_text = "Engine thinking..."
            else:
                self.state.turn_phase = TurnPhase.OPP_TURN
                self.state.status_text = "Your move"

        elif self.state.mode == GameMode.AI_VS_AI:
            self.state.turn_phase = TurnPhase.MY_TURN
            self._kick_engine_search()
            self.state.status_text = "Engine thinking..."

        return True

    def apply_engine_move(self, move_str: str) -> bool:
        """Apply a bestmove the engine emitted. Returns True if applied."""
        try:
            move = self._move_from_str(move_str)
        except (ValueError, KeyError):
            return False
        if not self._is_legal(move):
            return False
        self.board.push(move)
        self.state.move_history.append(move_str)
        self._send_position()
        self.state.status_text = f"Engine played: {move_str}"
        return True

    def _kick_engine_search(self) -> None:
        """Start the engine's bestmove search with standard time controls."""
        self._client.send_uci(
            "go wtime 600000 btime 600000 winc 5000 binc 5000"
        )

    # -----------------------------------------------------------------------
    # Undo
    # -----------------------------------------------------------------------

    def can_undo(self) -> bool:
        """True iff at least one ply can be undone in the current mode."""
        if not self.state.move_history:
            return False
        if self.state.mode == GameMode.SINGLE_SIDE:
            return self._compute_undo_count_single_side() > 0
        return True

    def undo(self) -> bool:
        """Mode-aware undo. Returns True if any ply was popped.

        - ANALYSIS / STANDARD: pop 1 ply, optionally restart analysis.
        - SINGLE_SIDE (Human vs AI): pop until it's the human's turn so the
          engine doesn't auto-reply. Refuses if that's not reachable.
        - AI_VS_AI: stop the engine, pop 2 plies, switch to STANDARD so the
          AI doesn't immediately think again. Press AI vs AI to resume.
        """
        if not self.can_undo():
            return False
        self._client.send_uci("stop")

        if self.state.mode == GameMode.SINGLE_SIDE:
            n = self._compute_undo_count_single_side()
            del self.state.move_history[-n:]
            self._rebuild_board()
            self.state.turn_phase = TurnPhase.OPP_TURN
            self.state.status_text = "Undo — your move"
        elif self.state.mode == GameMode.AI_VS_AI:
            n = min(2, len(self.state.move_history))
            del self.state.move_history[-n:]
            self._rebuild_board()
            self.state.mode = GameMode.STANDARD
            self.state.analysis_active = False
            self.state.turn_phase = TurnPhase.IDLE
            self.state.status_text = "AI vs AI paused — undo applied"
        else:
            self.state.move_history.pop()
            self.board.pop()
            self.state.status_text = "Undo"

        self._latest_info.clear()
        self._send_position()
        if self.state.mode == GameMode.ANALYSIS and self.state.analysis_active:
            self._start_analysis()
        return True

    def _rebuild_board(self) -> None:
        """Rebuild self.board from move_history (single source of truth)."""
        self.board = self._new_board()
        for s in self.state.move_history:
            self.board.push(self._move_from_str(s))

    def _compute_undo_count_single_side(self) -> int:
        """How many plies to pop in SINGLE_SIDE so it's the human's turn.

        Returns 0 when no such count exists (e.g. engine is the first mover
        and only its opening move has been played).
        """
        current = int(self.board.turn)
        my = int(self.state.my_side)
        for k in range(1, len(self.state.move_history) + 1):
            if (current ^ (k & 1)) != my:
                return k
        return 0

    # -----------------------------------------------------------------------
    # Mode switching
    # -----------------------------------------------------------------------

    def set_mode(self, mode: GameMode) -> None:
        """Switch game mode (idempotent on identical mode)."""
        if mode == self.state.mode:
            return
        self._client.send_uci("stop")
        self.state.mode = mode
        self.state.status_text = f"Mode: {mode.name}"

        if mode == GameMode.ANALYSIS:
            self._client.send_uci(f"setoption name MultiPV value {self._multi_pv}")
            self.state.analysis_active = True
            self._start_analysis()
        elif mode == GameMode.AI_VS_AI:
            self.state.analysis_active = False
            self._send_position()
            self._kick_engine_search()
            self.state.status_text = "AI vs AI — engine thinking..."
        else:
            self.state.analysis_active = False

    def set_single_side(self, engine_side: int) -> None:
        """Switch to SINGLE_SIDE with the engine playing *engine_side*.

        Idempotent on (mode, side). When entering the mode, kicks off whichever
        phase applies for the current position.
        """
        side_changed = self.state.my_side != engine_side
        mode_changed = self.state.mode != GameMode.SINGLE_SIDE
        if not side_changed and not mode_changed:
            return

        self._client.send_uci("stop")
        self.state.mode = GameMode.SINGLE_SIDE
        self.state.my_side = engine_side
        self.state.analysis_active = False

        self._client.send_uci(f"setoption name MultiPV value {self._multi_pv}")
        self.state.status_text = (
            f"Single mode — engine plays {_side_label(self._game, engine_side)}"
        )

        if self.board.turn == engine_side:
            self.state.turn_phase = TurnPhase.MY_TURN
            self._send_position()
            self._kick_engine_search()
        else:
            self.state.turn_phase = TurnPhase.OPP_TURN

    def flip_board(self) -> None:
        """Toggle board orientation (moves engine's side)."""
        # The in-chess representation of color inversion differs per game, but
        # both libraries treat ! as boolean inversion on a 0/1 value.
        self.state.my_side = int(not self.state.my_side)

    def toggle_analysis(self) -> None:
        """Toggle infinite analysis (see module docstring)."""
        if self.state.mode != GameMode.ANALYSIS:
            self.state.mode = GameMode.ANALYSIS
            self._client.send_uci(f"setoption name MultiPV value {self._multi_pv}")
            self.state.analysis_active = True
            self._start_analysis()
            return

        if self.state.analysis_active:
            self._client.send_uci("stop")
            self.state.analysis_active = False
            self.state.status_text = "Analysis paused"
        else:
            self.state.analysis_active = True
            self._start_analysis()

    def _start_analysis(self) -> None:
        self._client.send_uci("stop")
        self._send_position()
        self._client.send_uci("go infinite")
        self.state.status_text = "Analysing..."

    # -----------------------------------------------------------------------
    # Engine event processing
    # -----------------------------------------------------------------------

    def process_event(self, event: EngineEvent) -> bool:
        """Dispatch an engine event to the appropriate handler.

        Returns False when the event was a stale InfoUpdate that belongs to a
        search the engine started before the most recent position change —
        callers should propagate that decision (e.g. skip panel updates) so
        the UI doesn't render numbers for a position that no longer exists.
        For every other event type, returns True.
        """
        match event:
            case UciOkEvent(engine_name=name):
                self.state.engine_name = name or "Engine"
                self.state.is_engine_ready = True
                self.state.status_text = f"Connected to {self.state.engine_name}"

            case ReadyOkEvent():
                self.state.is_engine_ready = True
                if self._isready_outstanding > 0:
                    self._isready_outstanding -= 1

            case InfoUpdate() as info:
                if self._isready_outstanding > 0:
                    return False  # stale, predates the latest position change
                self._latest_info[info.multipv] = info
                if info.multipv == 1:
                    self._maybe_set_mate_status(info)

            case BestMoveEvent(move=move_str):
                self._handle_bestmove(move_str)

            case EngineError(message=msg):
                self.state.status_text = f"Engine error: {msg[:60]}"

            case _:
                pass

        return True

    def _handle_bestmove(self, move_str: str) -> None:
        # Defense in depth: if the latest top-1 was a forced mate but the
        # bestmove the engine returned isn't on that PV, surface a warning.
        # The engine fix in mcts.hpp should prevent this in normal operation,
        # but older builds or unforeseen races may still trip it.
        warn = self._mate_pv_mismatch(move_str)

        if self.state.mode == GameMode.SINGLE_SIDE:
            # Accept the bestmove if the controller asked the engine to think
            # (MY_TURN), or if it's the engine's turn on the board even if our
            # turn_phase tracking lagged (e.g. fresh new_game where the engine
            # plays first). Otherwise it's a stale bestmove from a prior
            # search after stop/undo and must be dropped.
            engine_turn_on_board = self.board.turn == self.state.my_side
            if self.state.turn_phase == TurnPhase.MY_TURN or engine_turn_on_board:
                if self.apply_engine_move(move_str):
                    self.state.turn_phase = TurnPhase.OPP_TURN
                    self.state.status_text = (
                        warn or "Your move"
                    )
                else:
                    self.state.status_text = (
                        f"⚠ Engine returned illegal/null move {move_str!r}"
                    )
            else:
                self.state.status_text = (
                    f"(stale bestmove {move_str} ignored)"
                )
        elif self.state.mode == GameMode.AI_VS_AI:
            # Apply the engine's move and immediately schedule the next search.
            if self.apply_engine_move(move_str):
                if self.board.is_game_over():
                    self.state.status_text = "Game over"
                    self.state.turn_phase = TurnPhase.IDLE
                else:
                    self._latest_info.clear()
                    self._kick_engine_search()
                    if warn:
                        self.state.status_text = warn

    def _mate_pv_mismatch(self, move_str: str) -> str | None:
        info = self._latest_info.get(1)
        if info is None or not info.pv:
            return None
        is_mate = (
            info.score_mate is not None
            or info.score_cp >= 29000
            or info.score_cp <= -29000
        )
        if not is_mate:
            return None
        if info.pv[0] == move_str:
            return None
        return f"⚠ Engine bestmove {move_str} is not on the mate PV ({info.pv[0]})"

    def _maybe_set_mate_status(self, info: InfoUpdate) -> None:
        if info.score_mate is not None:
            n = abs(info.score_mate)
            label = "必勝" if info.score_mate > 0 else "必敗"
            self.state.status_text = f"{label} — Mate in {n}"
            return
        if info.score_cp >= 29000:
            n = max(1, (32000 - info.score_cp) // 2 + 1)
            self.state.status_text = f"必勝 — Mate in {n}"
        elif info.score_cp <= -29000:
            n = max(1, (32000 + info.score_cp) // 2 + 1)
            self.state.status_text = f"必敗 — Mate in {n}"

    def latest_infos(self) -> list[InfoUpdate]:
        return [self._latest_info[k] for k in sorted(self._latest_info)]
