"""Game scene — in-game UI and event loop, driven by a `GameConfig`.

This module owns the pygame widgets that make up the playing screen (board,
eval bar, analysis panel, move list, status bar, button bar) and dispatches
input events. The menu scene hands a `GameConfig` here; the scene returns a
`GameSceneAction` (e.g. BACK_TO_MENU, QUIT) when the user chooses to leave.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import chess
import pygame
import shogi

from gui.analysis_panel import AnalysisPanel
from gui.chess_board import ChessBoardRenderer, MoveArrow
from gui.engine_client import EngineClient, InfoUpdate
from gui.game_controller import GameController, GameMode, TurnPhase
from gui.menu import GameConfig
from gui.shogi_board import PendingPromotion, ShogiBoardRenderer
from gui.themes import (
    BOARD_MARGIN,
    BUTTON_BAR_H,
    DEFAULT_THEME,
    EVAL_BAR_W,
    FONT_SMALL,
    MOVE_LIST_H,
    RIGHT_PANEL_W,
    STATUS_BAR_H,
    Theme,
)


class GameSceneAction(Enum):
    NONE = auto()
    BACK_TO_MENU = auto()
    QUIT = auto()


# Button keys — stable identifiers independent of user-facing labels.
BTN_ANALYSIS = "analysis"
BTN_AI_AI = "ai_ai"
BTN_STANDARD = "standard"
BTN_FLIP = "flip"
BTN_UNDO = "undo"
BTN_NEW = "new"
BTN_MENU = "menu"


def compute_layout(w: int, h: int) -> dict[str, pygame.Rect]:
    """Compute panel rects for given window size."""
    rp_w = RIGHT_PANEL_W
    eb_w = EVAL_BAR_W + BOARD_MARGIN * 2

    board_left = eb_w
    board_right = w - rp_w
    board_top = 0
    board_bottom = h - BUTTON_BAR_H - STATUS_BAR_H - MOVE_LIST_H

    board_area_w = board_right - board_left
    board_area_h = board_bottom - board_top

    board_side = min(board_area_w, board_area_h)
    board_x = board_left + (board_area_w - board_side) // 2
    board_y = board_top + (board_area_h - board_side) // 2

    board_rect = pygame.Rect(board_x, board_y, board_side, board_side)

    eval_bar_rect = pygame.Rect(
        BOARD_MARGIN, board_y, EVAL_BAR_W, board_side
    )
    right_panel_rect = pygame.Rect(w - rp_w, 0, rp_w, h)
    move_list_rect = pygame.Rect(
        0, h - BUTTON_BAR_H - STATUS_BAR_H - MOVE_LIST_H, w - rp_w, MOVE_LIST_H
    )
    status_bar_rect = pygame.Rect(
        0, h - BUTTON_BAR_H - STATUS_BAR_H, w - rp_w, STATUS_BAR_H
    )
    button_bar_rect = pygame.Rect(0, h - BUTTON_BAR_H, w - rp_w, BUTTON_BAR_H)

    return {
        "board": board_rect,
        "eval_bar": eval_bar_rect,
        "right_panel": right_panel_rect,
        "move_list": move_list_rect,
        "status_bar": status_bar_rect,
        "button_bar": button_bar_rect,
    }


# -------------------------------------------------------------------------
# Simple clickable button (moved from app.py)
# -------------------------------------------------------------------------


class Button:
    """Clickable button with optional sticky `selected` highlight."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        font: pygame.font.Font,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        self.rect = rect
        self.label = label
        self.font = font
        self.theme = theme
        self.selected = False
        self._hovered = False
        self._pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._pressed and self.rect.collidepoint(event.pos):
                self._pressed = False
                return True
            self._pressed = False
        return False

    def draw(self, surface: pygame.Surface) -> None:
        if self._pressed or self.selected:
            color = self.theme.btn_active
        elif self._hovered:
            color = self.theme.btn_hover
        else:
            color = self.theme.btn_normal
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, self.theme.panel_border, self.rect, 1, border_radius=4)
        text_surf = self.font.render(self.label, True, self.theme.btn_text)
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))


# -------------------------------------------------------------------------
# Move list + status bar rendering
# -------------------------------------------------------------------------


def _render_chess_move_list(
    surface: pygame.Surface,
    rect: pygame.Rect,
    move_history: list[str],
    theme: Theme,
) -> None:
    pygame.draw.rect(surface, theme.move_list_bg, rect)
    pygame.draw.rect(surface, theme.panel_border, rect, 1)

    font = pygame.font.SysFont("sans", FONT_SMALL)
    x_margin = rect.left + 6
    y = rect.top + 4
    line_h = font.get_height() + 2
    col_w = rect.width // 2 - 10

    b = chess.Board()
    san_moves: list[str] = []
    for uci_str in move_history:
        try:
            move = chess.Move.from_uci(uci_str)
            san_moves.append(b.san(move))
            b.push(move)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            san_moves.append(uci_str)

    for i in range(0, len(san_moves), 2):
        move_num = i // 2 + 1
        num_surf = font.render(f"{move_num}.", True, theme.move_number)
        surface.blit(num_surf, (x_margin, y))
        white_str = san_moves[i] if i < len(san_moves) else ""
        white_surf = font.render(white_str, True, theme.move_white)
        surface.blit(white_surf, (x_margin + 28, y))
        if i + 1 < len(san_moves):
            black_str = san_moves[i + 1]
            black_surf = font.render(black_str, True, theme.move_black)
            surface.blit(black_surf, (x_margin + col_w, y))
        y += line_h
        if y > rect.bottom - line_h:
            break


def _render_shogi_move_list(
    surface: pygame.Surface,
    rect: pygame.Rect,
    move_history: list[str],
    theme: Theme,
) -> None:
    """Two-column list of USI moves (no SAN rebuild — faster and correct)."""
    pygame.draw.rect(surface, theme.move_list_bg, rect)
    pygame.draw.rect(surface, theme.panel_border, rect, 1)
    font = pygame.font.SysFont("sans", FONT_SMALL)
    x_margin = rect.left + 6
    y = rect.top + 4
    line_h = font.get_height() + 2
    col_w = rect.width // 2 - 10

    for i in range(0, len(move_history), 2):
        move_num = i // 2 + 1
        num_surf = font.render(f"{move_num}.", True, theme.move_number)
        surface.blit(num_surf, (x_margin, y))
        sente_str = move_history[i] if i < len(move_history) else ""
        sente_surf = font.render(sente_str, True, theme.move_white)
        surface.blit(sente_surf, (x_margin + 28, y))
        if i + 1 < len(move_history):
            gote_surf = font.render(move_history[i + 1], True, theme.move_black)
            surface.blit(gote_surf, (x_margin + col_w, y))
        y += line_h
        if y > rect.bottom - line_h:
            break


def _render_status_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    status: str,
    theme: Theme,
) -> None:
    pygame.draw.rect(surface, theme.panel_header, rect)
    pygame.draw.rect(surface, theme.panel_border, rect, 1)
    font = pygame.font.SysFont("sans", FONT_SMALL)
    surf = font.render(status, True, theme.text_primary)
    surface.blit(surf, surf.get_rect(midleft=(rect.left + 8, rect.centery)))


# -------------------------------------------------------------------------
# The game scene itself
# -------------------------------------------------------------------------


@dataclass
class _ChessPendingPromotion:
    """Chess promotion that needs a piece-type choice."""

    from_square: int
    to_square: int


class GameScene:
    """Owns the in-game state + widgets and drives the main-loop frame."""

    def __init__(self, screen: pygame.Surface, config: GameConfig) -> None:
        self.screen = screen
        self.config = config
        self.theme = DEFAULT_THEME
        self._flipped = config.play_mode == "human_ai" and config.human_side == "black"

        # Engine + controller
        self._client = EngineClient(config.engine_path_for_client(), config.game)
        self._controller = GameController(
            client=self._client,
            game=config.game,
            mode=self._initial_mode(),
            my_side=self._initial_engine_side(),
        )
        self._client.start()
        self._controller.initialize_engine()
        self._client.wait_for_uciok(timeout=3.0)
        self._controller.new_game()

        # Layout + renderers
        self._layout = compute_layout(*screen.get_size())
        if config.game == "chess":
            self._board_renderer: Any = ChessBoardRenderer(
                surface=screen,
                rect=self._layout["board"],
                board=self._controller.board,
                theme=self.theme,
                flipped=self._flipped,
            )
        else:
            self._board_renderer = ShogiBoardRenderer(
                surface=screen,
                rect=self._layout["board"],
                board=self._controller.board,
                theme=self.theme,
                flipped=self._flipped,
            )

        self._analysis_panel = AnalysisPanel(
            surface=screen,
            right_panel_rect=self._layout["right_panel"],
            eval_bar_rect=self._layout["eval_bar"],
            theme=self.theme,
        )

        # UI state
        self._btn_font = pygame.font.SysFont("sans", FONT_SMALL)
        self._buttons = self._build_buttons()
        self._selected_square: int | None = None
        self._chess_pending_promo: _ChessPendingPromotion | None = None
        self._chess_promo_buttons: dict[str, pygame.Rect] = {}

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def _initial_mode(self) -> GameMode:
        if self.config.play_mode == "ai_ai":
            return GameMode.AI_VS_AI
        if self.config.play_mode == "human_ai":
            return GameMode.SINGLE_SIDE
        return GameMode.STANDARD

    def _initial_engine_side(self) -> int:
        """Return the side the engine plays given the player's chosen human side."""
        if self.config.play_mode != "human_ai":
            # Defaults don't really matter for other modes.
            return (
                chess.WHITE if self.config.game == "chess" else shogi.BLACK
            )
        # Engine plays opposite of the human.
        if self.config.human_side == "white":
            return chess.BLACK if self.config.game == "chess" else shogi.WHITE
        return chess.WHITE if self.config.game == "chess" else shogi.BLACK

    def _human_can_move(self) -> bool:
        """True when the local human input is allowed to move the piece on turn."""
        mode = self._controller.state.mode
        if mode == GameMode.STANDARD:
            return True
        if mode == GameMode.ANALYSIS:
            return True  # let the user freely manipulate the board
        if mode == GameMode.SINGLE_SIDE:
            return self._controller.board.turn != self._controller.state.my_side
        return False  # AI_VS_AI: human is a spectator

    # ------------------------------------------------------------------
    # Button bar
    # ------------------------------------------------------------------

    def _build_buttons(self) -> list[tuple[str, Button]]:
        bar = self._layout["button_bar"]
        specs = [
            (BTN_MENU, "< Menu"),
            (BTN_ANALYSIS, "Analysis"),
            (BTN_AI_AI, "AI vs AI"),
            (BTN_STANDARD, "Standard"),
            (BTN_FLIP, "Flip"),
            (BTN_UNDO, "Undo"),
            (BTN_NEW, "New"),
        ]
        bw = 76
        bh = bar.height - 8
        bx = bar.left + 6
        buttons: list[tuple[str, Button]] = []
        for key, lbl in specs:
            r = pygame.Rect(bx, bar.top + 4, bw, bh)
            buttons.append((key, Button(r, lbl, self._btn_font, self.theme)))
            bx += bw + 4
        return buttons

    def _update_button_selected(self) -> None:
        state = self._controller.state
        analysing = (
            state.mode == GameMode.ANALYSIS and state.analysis_active
        )
        for key, btn in self._buttons:
            if key == BTN_ANALYSIS:
                btn.selected = analysing
            elif key == BTN_AI_AI:
                btn.selected = state.mode == GameMode.AI_VS_AI
            elif key == BTN_STANDARD:
                btn.selected = state.mode == GameMode.STANDARD
            else:
                btn.selected = False

    def _handle_button_click(self, key: str) -> GameSceneAction:
        controller = self._controller
        if key == BTN_MENU:
            return GameSceneAction.BACK_TO_MENU
        if key == BTN_ANALYSIS:
            controller.toggle_analysis()
        elif key == BTN_AI_AI:
            controller.set_mode(GameMode.AI_VS_AI)
        elif key == BTN_STANDARD:
            controller.set_mode(GameMode.STANDARD)
        elif key == BTN_FLIP:
            self._flipped = not self._flipped
            self._board_renderer.flipped = self._flipped
        elif key == BTN_UNDO:
            self._do_undo()
        elif key == BTN_NEW:
            controller.new_game()
            self._clear_selection()
            self._analysis_panel.reset()
            if isinstance(self._board_renderer, ChessBoardRenderer):
                self._board_renderer.set_selected(None)
                self._board_renderer.set_last_move(None)
                self._board_renderer.set_arrows([])
            else:
                self._board_renderer.clear_selection()
                self._board_renderer.set_last_move(None)
                self._board_renderer.set_pending_promotion(None)
        return GameSceneAction.NONE

    def _do_undo(self) -> None:
        if not self._controller.undo():
            return
        self._clear_selection()
        self._analysis_panel.reset()
        if isinstance(self._board_renderer, ChessBoardRenderer):
            self._board_renderer.set_selected(None)
            self._board_renderer.set_last_move(None)
            self._board_renderer.set_arrows([])
        else:
            self._board_renderer.clear_selection()
            self._board_renderer.set_last_move(None)
            self._board_renderer.set_pending_promotion(None)

    def _clear_selection(self) -> None:
        self._selected_square = None
        self._chess_pending_promo = None
        if isinstance(self._board_renderer, ShogiBoardRenderer):
            self._board_renderer.clear_selection()

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def on_resize(self, size: tuple[int, int]) -> None:
        self._layout = compute_layout(*size)
        self._board_renderer.surface = self.screen
        self._board_renderer.rect = self._layout["board"]
        self._analysis_panel.surface = self.screen
        self._analysis_panel.right_panel_rect = self._layout["right_panel"]
        self._analysis_panel.eval_bar_rect = self._layout["eval_bar"]
        self._buttons = self._build_buttons()

    def handle_event(self, event: pygame.event.Event) -> GameSceneAction:
        if event.type == pygame.QUIT:
            return GameSceneAction.QUIT

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return GameSceneAction.BACK_TO_MENU
            if event.key == pygame.K_f:
                self._flipped = not self._flipped
                self._board_renderer.flipped = self._flipped
            elif event.key == pygame.K_n:
                self._controller.new_game()
                self._clear_selection()
                self._analysis_panel.reset()
            elif event.key == pygame.K_LEFT:
                self._do_undo()

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Promotion dialog intercepts all clicks until resolved.
            if self._chess_pending_promo is not None:
                self._handle_chess_promo_click(event.pos)
                return GameSceneAction.NONE
            if isinstance(self._board_renderer, ShogiBoardRenderer):
                if self._board_renderer.pending_promotion is not None:
                    self._handle_shogi_promo_click(event.pos)
                    return GameSceneAction.NONE

            if isinstance(self._board_renderer, ChessBoardRenderer):
                self._handle_chess_click(event.pos)
            else:
                self._handle_shogi_click(event.pos)

        # Button events are checked after board clicks so the board doesn't
        # absorb a click that landed on a button due to overlapping hitboxes.
        for key, btn in self._buttons:
            if btn.handle_event(event):
                result = self._handle_button_click(key)
                if result != GameSceneAction.NONE:
                    return result
        return GameSceneAction.NONE

    def process_engine_events(self) -> None:
        for eng_event in self._client.poll_events():
            self._controller.process_event(eng_event)
            if isinstance(eng_event, InfoUpdate):
                self._analysis_panel.update_info(eng_event)
                self._update_arrows()

    def render(self) -> None:
        theme = self.theme
        self.screen.fill(theme.window_bg)

        # Board
        if isinstance(self._board_renderer, ChessBoardRenderer):
            self._board_renderer.board = self._controller.board
        else:
            self._board_renderer.board = self._controller.board
        self._board_renderer.render()

        # Chess promotion overlay (drawn last so it's on top)
        if self._chess_pending_promo is not None:
            self._draw_chess_promotion_overlay()

        self._analysis_panel.render()

        if self.config.game == "chess":
            _render_chess_move_list(
                self.screen,
                self._layout["move_list"],
                self._controller.state.move_history,
                theme,
            )
        else:
            _render_shogi_move_list(
                self.screen,
                self._layout["move_list"],
                self._controller.state.move_history,
                theme,
            )

        _render_status_bar(
            self.screen,
            self._layout["status_bar"],
            self._controller.state.status_text,
            theme,
        )

        pygame.draw.rect(self.screen, theme.panel_bg, self._layout["button_bar"])
        pygame.draw.rect(self.screen, theme.panel_border, self._layout["button_bar"], 1)
        self._update_button_selected()
        for _, btn in self._buttons:
            btn.draw(self.screen)

    def cleanup(self) -> None:
        """Stop the engine client cleanly before scene teardown."""
        try:
            self._client.send_uci("stop")
        except Exception:
            pass
        self._client.stop()

    # ------------------------------------------------------------------
    # Chess click handling
    # ------------------------------------------------------------------

    def _handle_chess_click(self, pos: tuple[int, int]) -> None:
        if not self._human_can_move():
            return
        renderer = self._board_renderer
        assert isinstance(renderer, ChessBoardRenderer)
        sq = renderer.square_at_pixel(*pos)
        if sq is None:
            self._selected_square = None
            renderer.set_selected(None)
            return
        controller = self._controller
        piece = controller.board.piece_at(sq)

        if self._selected_square is None:
            if piece is not None and piece.color == controller.board.turn:
                self._selected_square = sq
                renderer.set_selected(sq)
            return

        # There is a selected square — attempt to move.
        from_sq = self._selected_square
        from_piece = controller.board.piece_at(from_sq)
        if from_piece is None:
            # Selection gone; treat as re-select
            self._selected_square = sq if piece and piece.color == controller.board.turn else None
            renderer.set_selected(self._selected_square)
            return

        # Is this a promotion pawn?
        is_promotion = (
            from_piece.piece_type == chess.PAWN
            and (
                (from_piece.color == chess.WHITE and chess.square_rank(sq) == 7)
                or (from_piece.color == chess.BLACK and chess.square_rank(sq) == 0)
            )
        )
        if is_promotion:
            # Check at least one promoted variant is legal before prompting.
            candidate = chess.Move(from_sq, sq, promotion=chess.QUEEN)
            if candidate in controller.board.legal_moves:
                self._chess_pending_promo = _ChessPendingPromotion(from_sq, sq)
                return

        move = chess.Move(from_sq, sq)
        if controller.try_move(move):
            renderer.set_last_move(move)
            renderer.set_selected(None)
            self._selected_square = None
            self._update_arrows()
        elif piece is not None and piece.color == controller.board.turn:
            self._selected_square = sq
            renderer.set_selected(sq)
        else:
            self._selected_square = None
            renderer.set_selected(None)

    def _handle_chess_promo_click(self, pos: tuple[int, int]) -> None:
        for piece_key, rect in self._chess_promo_buttons.items():
            if rect.collidepoint(pos):
                if piece_key == "cancel":
                    self._chess_pending_promo = None
                    return
                assert self._chess_pending_promo is not None
                promotion_type = {
                    "queen": chess.QUEEN,
                    "rook": chess.ROOK,
                    "bishop": chess.BISHOP,
                    "knight": chess.KNIGHT,
                }[piece_key]
                move = chess.Move(
                    self._chess_pending_promo.from_square,
                    self._chess_pending_promo.to_square,
                    promotion=promotion_type,
                )
                if self._controller.try_move(move):
                    self._board_renderer.set_last_move(move)
                    self._board_renderer.set_selected(None)
                    self._selected_square = None
                    self._update_arrows()
                self._chess_pending_promo = None
                return
        # Click outside the prompt — cancel.
        self._chess_pending_promo = None

    def _draw_chess_promotion_overlay(self) -> None:
        assert self._chess_pending_promo is not None
        # Dim the board
        dim = pygame.Surface(self._layout["board"].size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        self.screen.blit(dim, self._layout["board"].topleft)

        # Centered panel with Q/R/B/N choices
        panel_w, panel_h = 320, 120
        br = self._layout["board"]
        panel = pygame.Rect(
            br.centerx - panel_w // 2,
            br.centery - panel_h // 2,
            panel_w,
            panel_h,
        )
        pygame.draw.rect(self.screen, (30, 32, 42), panel, border_radius=10)
        pygame.draw.rect(self.screen, (160, 170, 200), panel, 2, border_radius=10)

        title_font = pygame.font.SysFont("sans", FONT_SMALL, bold=True)
        title = title_font.render("Promote pawn to:", True, (235, 235, 240))
        self.screen.blit(title, title.get_rect(midtop=(panel.centerx, panel.top + 8)))

        # Buttons
        from gui.chess_board import _load_chess_piece_font
        piece_font = _load_chess_piece_font(40)
        glyphs = {
            "queen": "♕" if self._promo_color_is_white() else "♛",
            "rook": "♖" if self._promo_color_is_white() else "♜",
            "bishop": "♗" if self._promo_color_is_white() else "♝",
            "knight": "♘" if self._promo_color_is_white() else "♞",
        }
        self._chess_promo_buttons = {}
        btn_w = 64
        btn_h = 64
        total_w = 4 * btn_w + 3 * 10
        x = panel.centerx - total_w // 2
        y = panel.top + 38
        for key in ("queen", "rook", "bishop", "knight"):
            rect = pygame.Rect(x, y, btn_w, btn_h)
            pygame.draw.rect(self.screen, (60, 65, 80), rect, border_radius=6)
            pygame.draw.rect(self.screen, (120, 130, 150), rect, 1, border_radius=6)
            glyph = piece_font.render(glyphs[key], True, (235, 235, 240))
            self.screen.blit(glyph, glyph.get_rect(center=rect.center))
            self._chess_promo_buttons[key] = rect
            x += btn_w + 10

    def _promo_color_is_white(self) -> bool:
        assert self._chess_pending_promo is not None
        piece = self._controller.board.piece_at(self._chess_pending_promo.from_square)
        return piece is not None and piece.color == chess.WHITE

    # ------------------------------------------------------------------
    # Shogi click handling
    # ------------------------------------------------------------------

    def _handle_shogi_click(self, pos: tuple[int, int]) -> None:
        if not self._human_can_move():
            return
        renderer = self._board_renderer
        assert isinstance(renderer, ShogiBoardRenderer)
        controller = self._controller

        # Did the user click a hand piece?
        hand_hit = renderer.hand_piece_at_pixel(*pos)
        if hand_hit is not None:
            color, ptype = hand_hit
            # Only the side to move can pick up its own hand pieces.
            if color != controller.board.turn:
                return
            renderer.set_selected_hand(color, ptype)
            self._selected_square = None
            return

        sq = renderer.square_at_pixel(*pos)
        if sq is None:
            renderer.clear_selection()
            self._selected_square = None
            return

        # Drop case: a hand piece is selected.
        if renderer.selected_hand is not None:
            color, ptype = renderer.selected_hand
            drop_move = renderer.build_candidate_drop(ptype, sq)
            if drop_move is not None:
                if controller.try_move(drop_move):
                    renderer.set_last_move(drop_move)
                    renderer.clear_selection()
                    self._update_arrows()
            else:
                renderer.clear_selection()
            return

        # Board-move case: either select or attempt move.
        piece = controller.board.piece_at(sq)
        if self._selected_square is None:
            if piece is not None and piece.color == controller.board.turn:
                self._selected_square = sq
                renderer.set_selected_square(sq)
            return

        from_sq = self._selected_square
        candidate = renderer.build_candidate_move(from_sq, sq)

        if candidate is None:
            # Not a legal move; maybe re-select.
            if piece is not None and piece.color == controller.board.turn:
                self._selected_square = sq
                renderer.set_selected_square(sq)
            else:
                self._selected_square = None
                renderer.clear_selection()
            return

        if isinstance(candidate, PendingPromotion):
            renderer.set_pending_promotion(candidate)
            return

        # Plain, unambiguous move.
        if controller.try_move(candidate):
            renderer.set_last_move(candidate)
            renderer.clear_selection()
            self._selected_square = None
            self._update_arrows()
        else:
            self._selected_square = None
            renderer.clear_selection()

    def _handle_shogi_promo_click(self, pos: tuple[int, int]) -> None:
        renderer = self._board_renderer
        assert isinstance(renderer, ShogiBoardRenderer)
        pending = renderer.pending_promotion
        if pending is None:
            return
        choice = renderer.promo_button_at_pixel(*pos)
        if choice is None:
            return
        move = pending.move_promote if choice == "promote" else pending.move_plain
        if self._controller.try_move(move):
            renderer.set_last_move(move)
            renderer.set_pending_promotion(None)
            renderer.clear_selection()
            self._selected_square = None
            self._update_arrows()

    # ------------------------------------------------------------------
    # Arrows
    # ------------------------------------------------------------------

    def _update_arrows(self) -> None:
        if not isinstance(self._board_renderer, ChessBoardRenderer):
            return
        arrows: list[MoveArrow] = []
        for info in self._controller.latest_infos()[:3]:
            if not info.pv:
                continue
            uci = info.pv[0]
            try:
                move = chess.Move.from_uci(uci)
                arrows.append(
                    MoveArrow(
                        from_sq=move.from_square,
                        to_sq=move.to_square,
                        rank=info.multipv,
                        score_cp=info.score_cp,
                    )
                )
            except ValueError:
                pass
        self._board_renderer.set_arrows(arrows)
