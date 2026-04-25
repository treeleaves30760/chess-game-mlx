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
    PolicyPreview,
    PonderHit,
    PonderMiss,
    PonderProgress,
    ReadyOkEvent,
    UciOkEvent,
)
from gui.ponder_manager import PonderManager


class GameMode(Enum):
    ANALYSIS = auto()         # go infinite, always analysing
    SINGLE_SIDE = auto()      # engine plays only one color; multi-ponder on opp turn
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
        ponder_manager: PonderManager,
        game: GameKind = "chess",
        mode: GameMode = GameMode.ANALYSIS,
        my_side: int | None = None,
    ) -> None:
        self._client = client
        self._ponder = ponder_manager
        self._game: GameKind = game
        self.board: Any = self._new_board()
        side = my_side if my_side is not None else _first_player(game)
        self.state = ControllerState(mode=mode, my_side=side)
        self._multi_pv = 3
        self._latest_info: dict[int, InfoUpdate] = {}

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
        self._client.send_uci("isready")

        if self.state.mode in (GameMode.ANALYSIS, GameMode.SINGLE_SIDE, GameMode.AI_VS_AI):
            self._client.send_uci(f"setoption name MultiPV value {self._multi_pv}")

        if self.state.mode == GameMode.SINGLE_SIDE:
            self._client.send_jsonrpc(
                "set_side", {"me": _side_label(self._game, self.state.my_side)}
            )

    def new_game(self) -> None:
        """Reset board and engine state."""
        self.board = self._new_board()
        self._ponder.reset()
        self._latest_info.clear()
        self.state.move_history = []
        self.state.status_text = self._initial_status()
        self.state.turn_phase = TurnPhase.IDLE

        self._client.send_uci(self._newgame_cmd())
        self._send_position()

        if self.state.mode == GameMode.ANALYSIS and self.state.analysis_active:
            self._start_analysis()
        elif self.state.mode == GameMode.AI_VS_AI:
            self._kick_engine_search()
        elif self.state.mode == GameMode.SINGLE_SIDE:
            if self.board.turn == self.state.my_side:
                self._kick_engine_search()

    def _initial_status(self) -> str:
        first = "White" if self._game == "chess" else "Sente (先手)"
        return f"New game — {first} to move"

    def _newgame_cmd(self) -> str:
        return "ucinewgame" if self._game == "chess" else "usinewgame"

    def _send_position(self) -> None:
        self._client.send_uci(self._position_cmd())

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
                self.state.status_text = "Waiting for opponent... (multi-ponder active)"
                self._client.send_jsonrpc("start_multi_ponder", {"k": 5})

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

    def opponent_played(self, move: Any) -> bool:
        """Inform controller that opponent played *move* (single-side mode)."""
        if not self._is_legal(move):
            return False
        self._client.send_jsonrpc("opponent_played", {"move": self._move_to_str(move)})
        self.board.push(move)
        self.state.move_history.append(self._move_to_str(move))
        self._send_position()
        return True

    def _kick_engine_search(self) -> None:
        """Start the engine's bestmove search with standard time controls."""
        self._client.send_uci(
            "go wtime 600000 btime 600000 winc 5000 binc 5000"
        )

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

        self._client.send_jsonrpc(
            "set_side", {"me": _side_label(self._game, engine_side)}
        )
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

    def process_event(self, event: EngineEvent) -> None:
        """Dispatch an engine event to the appropriate handler."""
        match event:
            case UciOkEvent(engine_name=name):
                self.state.engine_name = name or "Engine"
                self.state.is_engine_ready = True
                self.state.status_text = f"Connected to {self.state.engine_name}"

            case ReadyOkEvent():
                self.state.is_engine_ready = True

            case InfoUpdate() as info:
                self._latest_info[info.multipv] = info

            case BestMoveEvent(move=move_str):
                self._handle_bestmove(move_str)

            case PolicyPreview(moves=moves):
                self._ponder.on_policy_preview(moves)

            case PonderProgress(trees=trees):
                self._ponder.on_ponder_progress(trees)

            case PonderHit(tree=t, instant_bestmove=bm, score_cp=cp):
                self._ponder.on_ponder_hit(t, bm, cp)
                self.state.status_text = f"Ponder HIT! Best: {bm}"

            case PonderMiss(trees_discarded=n):
                self._ponder.on_ponder_miss(n)
                self.state.status_text = "Ponder miss — fresh search"

            case EngineError(message=msg):
                self.state.status_text = f"Engine error: {msg[:60]}"

            case _:
                pass

    def _handle_bestmove(self, move_str: str) -> None:
        if self.state.mode == GameMode.SINGLE_SIDE:
            if self.state.turn_phase == TurnPhase.MY_TURN:
                if self.apply_engine_move(move_str):
                    self.state.turn_phase = TurnPhase.OPP_TURN
                    self.state.status_text = "Opponent's turn"
        elif self.state.mode == GameMode.AI_VS_AI:
            # Apply the engine's move and immediately schedule the next search.
            if self.apply_engine_move(move_str):
                if self.board.is_game_over() if self._game == "chess" else self.board.is_game_over():
                    self.state.status_text = "Game over"
                    self.state.turn_phase = TurnPhase.IDLE
                else:
                    self._latest_info.clear()
                    self._kick_engine_search()

    def latest_infos(self) -> list[InfoUpdate]:
        return [self._latest_info[k] for k in sorted(self._latest_info)]
