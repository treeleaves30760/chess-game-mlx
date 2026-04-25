"""Home/menu scene for game setup.

Lets the user pick the game (chess/shogi), engine (rule-based or neural model
with a path), play mode (AI vs AI / Human vs AI / Human vs Human), and — when
relevant — which side the human takes. Returns a `GameConfig` to the main
application when the user clicks Start.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import pygame

from gui.themes import (
    DEFAULT_THEME,
    FONT_LARGE,
    FONT_MEDIUM,
    FONT_SMALL,
    FONT_TINY,
    MENU_CARD_GAP,
    MENU_CARD_H,
    MENU_CARD_W,
    MENU_SECTION_GAP,
    Theme,
)

# Fonts that can render both Latin and CJK (kanji for 将棋 / 先 / 後).
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "arialunicodems",       # macOS (Arial Unicode MS)
    "hiraginosansgb",       # macOS (Hiragino Sans GB)
    "hiraginosans",         # macOS (Hiragino Sans)
    "notosanscjkjp",        # Linux (Noto Sans CJK JP)
    "notosanscjksc",        # Linux (Noto Sans CJK SC)
    "msgothic",             # Windows (MS Gothic)
    "yugothic",             # Windows (Yu Gothic)
)


def _load_cjk_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Load a font covering Latin + CJK, verifying the resolved path exists.

    `pygame.font.match_font` can point to stale cache entries (e.g. /opt/X11
    paths that no longer exist on macOS), so probe each candidate before use.
    """
    available = set(pygame.font.get_fonts())
    for name in _CJK_FONT_CANDIDATES:
        if name not in available:
            continue
        path = pygame.font.match_font(name, bold=bold)
        if path and os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except OSError:
                continue
    return pygame.font.SysFont("sans", size, bold=bold)


GameKind = Literal["chess", "shogi"]
EngineKind = Literal["rule", "model"]
PlayMode = Literal["ai_ai", "human_ai", "human_human"]
HumanSide = Literal["white", "black"]


@dataclass
class GameConfig:
    """Bundle of choices passed from the menu into the game scene."""

    game: GameKind = "chess"
    engine_kind: EngineKind = "rule"
    model_path: str = ""
    play_mode: PlayMode = "human_ai"
    human_side: HumanSide = "white"

    @property
    def needs_engine(self) -> bool:
        return self.play_mode != "human_human"

    @property
    def needs_side_choice(self) -> bool:
        return self.play_mode == "human_ai"

    def engine_path_for_client(self) -> str:
        """Translate the menu choice into an EngineClient path argument."""
        if self.engine_kind == "model" and self.model_path:
            return self.model_path
        return "mock"


class MenuAction(Enum):
    NONE = auto()
    START = auto()
    QUIT = auto()


@dataclass
class _Card:
    rect: pygame.Rect
    label: str
    sublabel: str
    value: str
    group: str  # which selection group this belongs to


class MenuScene:
    """Renders and handles input for the main menu."""

    def __init__(self, surface: pygame.Surface, theme: Theme = DEFAULT_THEME) -> None:
        self.surface = surface
        self.theme = theme
        self.config = GameConfig()

        self._title_font = pygame.font.SysFont("sans", FONT_LARGE + 18, bold=True)
        self._subtitle_font = pygame.font.SysFont("sans", FONT_SMALL)
        self._section_font = pygame.font.SysFont("sans", FONT_SMALL, bold=True)
        # CJK-capable font for card titles — card labels include "将棋", "先", "後".
        self._card_title_font = _load_cjk_font(FONT_MEDIUM, bold=True)
        self._card_sub_font = pygame.font.SysFont("sans", FONT_TINY)
        self._input_font = pygame.font.SysFont("monospace", FONT_SMALL)
        self._start_font = pygame.font.SysFont("sans", FONT_MEDIUM + 4, bold=True)

        self._cards: list[_Card] = []
        self._start_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._input_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._hovered_rect: pygame.Rect | None = None
        self._hovered_start: bool = False
        self._input_focused: bool = False
        self._cursor_ms: int = 0

        self._build_layout(surface.get_size())

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self, size: tuple[int, int]) -> None:
        """Build card and button rects centered in the window."""
        w, h = size
        self._cards = []

        cx = w // 2
        top = 130

        # GAME row (2 cards)
        self._add_row(
            cx,
            top,
            group="game",
            items=[
                ("chess", "Chess", "8×8 · python-chess"),
                ("shogi", "Shogi 将棋", "9×9 · python-shogi"),
            ],
        )
        y_after_game = top + MENU_CARD_H + MENU_SECTION_GAP + 28

        # ENGINE row (2 cards)
        self._add_row(
            cx,
            y_after_game,
            group="engine",
            items=[
                ("rule", "Rule-based", "Built-in heuristic"),
                ("model", "Neural Model", "Load .safetensors"),
            ],
        )
        y_after_engine = y_after_game + MENU_CARD_H + 14

        # Model-path input row (always rendered, greyed when engine != model)
        input_w = MENU_CARD_W * 2 + MENU_CARD_GAP
        self._input_rect = pygame.Rect(
            cx - input_w // 2,
            y_after_engine,
            input_w,
            30,
        )
        y_after_input = y_after_engine + 30 + MENU_SECTION_GAP + 28

        # PLAY MODE row (3 cards)
        self._add_row(
            cx,
            y_after_input,
            group="mode",
            items=[
                ("ai_ai", "AI vs AI", "Watch engines play"),
                ("human_ai", "Human vs AI", "Play against engine"),
                ("human_human", "Human vs Human", "Local two-player"),
            ],
        )
        y_after_mode = y_after_input + MENU_CARD_H + MENU_SECTION_GAP + 28

        # SIDE row (2 cards, conditionally enabled for human_ai)
        self._add_row(
            cx,
            y_after_mode,
            group="side",
            items=[
                ("white", "White / 先", "You move first"),
                ("black", "Black / 後", "Engine opens"),
            ],
        )
        y_after_side = y_after_mode + MENU_CARD_H + MENU_SECTION_GAP + 12

        # Start button
        start_w, start_h = 220, 52
        self._start_rect = pygame.Rect(
            cx - start_w // 2,
            min(y_after_side, h - start_h - 30),
            start_w,
            start_h,
        )

    def _add_row(
        self,
        cx: int,
        y: int,
        group: str,
        items: list[tuple[str, str, str]],
    ) -> None:
        """Add a centered row of equally-sized cards."""
        n = len(items)
        total_w = n * MENU_CARD_W + (n - 1) * MENU_CARD_GAP
        x = cx - total_w // 2
        for value, label, sublabel in items:
            rect = pygame.Rect(x, y, MENU_CARD_W, MENU_CARD_H)
            self._cards.append(_Card(rect, label, sublabel, value, group))
            x += MENU_CARD_W + MENU_CARD_GAP

    def on_resize(self, size: tuple[int, int]) -> None:
        self._build_layout(size)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> MenuAction:
        if event.type == pygame.QUIT:
            return MenuAction.QUIT

        if event.type == pygame.MOUSEMOTION:
            pos = event.pos
            self._hovered_rect = None
            for card in self._cards:
                if self._card_enabled(card) and card.rect.collidepoint(pos):
                    self._hovered_rect = card.rect
                    break
            self._hovered_start = self._start_rect.collidepoint(pos)
            return MenuAction.NONE

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            for card in self._cards:
                if self._card_enabled(card) and card.rect.collidepoint(pos):
                    self._select_card(card)
                    return MenuAction.NONE
            if self._input_rect.collidepoint(pos):
                self._input_focused = True
                return MenuAction.NONE
            if self._start_rect.collidepoint(pos):
                return MenuAction.START
            self._input_focused = False
            return MenuAction.NONE

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return MenuAction.QUIT
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return MenuAction.START
            if self._input_focused and self.config.engine_kind == "model":
                self._handle_text_input(event)
            return MenuAction.NONE

        return MenuAction.NONE

    def _handle_text_input(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_BACKSPACE:
            self.config.model_path = self.config.model_path[:-1]
        elif event.key == pygame.K_v and (event.mod & pygame.KMOD_META or event.mod & pygame.KMOD_CTRL):
            try:
                clip = pygame.scrap.get(pygame.SCRAP_TEXT)
                if clip:
                    self.config.model_path += clip.decode("utf-8", "ignore").rstrip("\x00")
            except (pygame.error, AttributeError):
                pass
        elif event.unicode and event.unicode.isprintable():
            self.config.model_path += event.unicode

    def _select_card(self, card: _Card) -> None:
        if card.group == "game":
            self.config.game = card.value  # type: ignore[assignment]
        elif card.group == "engine":
            self.config.engine_kind = card.value  # type: ignore[assignment]
            self._input_focused = card.value == "model"
        elif card.group == "mode":
            self.config.play_mode = card.value  # type: ignore[assignment]
        elif card.group == "side":
            self.config.human_side = card.value  # type: ignore[assignment]

    def _card_enabled(self, card: _Card) -> bool:
        if card.group == "side" and not self.config.needs_side_choice:
            return False
        if card.group == "engine" and not self.config.needs_engine:
            return False
        return True

    def _card_selected(self, card: _Card) -> bool:
        if card.group == "game":
            return self.config.game == card.value
        if card.group == "engine":
            return self.config.engine_kind == card.value
        if card.group == "mode":
            return self.config.play_mode == card.value
        if card.group == "side":
            return self.config.human_side == card.value
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, dt_ms: int) -> None:
        self._cursor_ms = (self._cursor_ms + dt_ms) % 1000
        screen = self.surface
        w, h = screen.get_size()

        # Gradient background (vertical, subtle)
        self._draw_gradient_background(w, h)

        # Title + subtitle
        self._draw_title(w)

        # Section labels, rendered above their rows
        self._draw_section_labels()

        # Cards
        for card in self._cards:
            self._draw_card(card)

        # Model path input
        self._draw_input()

        # Start button
        self._draw_start_button()

        # Footer hint
        hint = self._subtitle_font.render(
            "Enter = Start · Esc = Quit",
            True,
            self.theme.text_secondary,
        )
        screen.blit(hint, hint.get_rect(midbottom=(w // 2, h - 12)))

    def _draw_gradient_background(self, w: int, h: int) -> None:
        top_c = (26, 30, 38)
        bot_c = (18, 20, 28)
        # Painter's gradient: fill once then paint horizontal bands. 40 bands
        # is visually smooth and cheap at any reasonable window size.
        bands = 40
        band_h = h // bands + 1
        for i in range(bands):
            t = i / (bands - 1)
            c = tuple(int(top_c[k] * (1 - t) + bot_c[k] * t) for k in range(3))
            pygame.draw.rect(self.surface, c, pygame.Rect(0, i * band_h, w, band_h))

    def _draw_title(self, w: int) -> None:
        title = self._title_font.render(
            "Chess + Shogi AI",
            True,
            (235, 235, 240),
        )
        self.surface.blit(title, title.get_rect(midtop=(w // 2, 42)))
        subtitle = self._subtitle_font.render(
            "MacBook Air M3 · MLX",
            True,
            self.theme.text_secondary,
        )
        self.surface.blit(
            subtitle,
            subtitle.get_rect(midtop=(w // 2, 42 + self._title_font.get_height() + 2)),
        )

    def _draw_section_labels(self) -> None:
        # Groups appear in card list order; pull the topmost y per group.
        groups = ["game", "engine", "mode", "side"]
        titles = {
            "game": "GAME",
            "engine": "ENGINE",
            "mode": "PLAY MODE",
            "side": "HUMAN PLAYS AS",
        }
        for group in groups:
            cards = [c for c in self._cards if c.group == group]
            if not cards:
                continue
            y = cards[0].rect.top - 22
            cx = (cards[0].rect.left + cards[-1].rect.right) // 2
            enabled = self._card_enabled(cards[0])
            color = (
                self.theme.text_primary if enabled else self.theme.text_secondary
            )
            surf = self._section_font.render(titles[group], True, color)
            self.surface.blit(surf, surf.get_rect(midbottom=(cx, y + 18)))

    def _draw_card(self, card: _Card) -> None:
        enabled = self._card_enabled(card)
        selected = enabled and self._card_selected(card)
        hovered = enabled and self._hovered_rect == card.rect

        if selected:
            bg = (58, 95, 140)
            border = (120, 180, 230)
            border_w = 2
        elif hovered:
            bg = (55, 60, 72)
            border = (120, 130, 150)
            border_w = 1
        else:
            bg = (44, 48, 58)
            border = (70, 75, 90)
            border_w = 1

        if not enabled:
            bg = (36, 38, 46)
            border = (60, 62, 72)

        pygame.draw.rect(self.surface, bg, card.rect, border_radius=8)
        pygame.draw.rect(self.surface, border, card.rect, border_w, border_radius=8)

        title_color = (230, 230, 235) if enabled else (120, 124, 134)
        sub_color = self.theme.text_secondary if enabled else (90, 94, 104)

        title = self._card_title_font.render(card.label, True, title_color)
        sub = self._card_sub_font.render(card.sublabel, True, sub_color)

        cx = card.rect.centerx
        ty = card.rect.top + 20
        self.surface.blit(title, title.get_rect(midtop=(cx, ty)))
        self.surface.blit(
            sub,
            sub.get_rect(midtop=(cx, ty + title.get_height() + 4)),
        )

    def _draw_input(self) -> None:
        active = self.config.engine_kind == "model"
        bg = (30, 32, 40) if active else (26, 28, 34)
        border = (
            (120, 180, 230)
            if active and self._input_focused
            else (70, 75, 90)
            if active
            else (50, 52, 60)
        )
        pygame.draw.rect(self.surface, bg, self._input_rect, border_radius=6)
        pygame.draw.rect(self.surface, border, self._input_rect, 1, border_radius=6)

        text_color = (220, 220, 230) if active else (100, 104, 114)
        placeholder = "/path/to/model.safetensors"
        display = self.config.model_path if self.config.model_path else placeholder
        color = text_color if self.config.model_path else (110, 114, 124)

        surf = self._input_font.render(display, True, color)
        inner = self._input_rect.inflate(-14, 0)
        self.surface.blit(
            surf,
            surf.get_rect(midleft=(inner.left, self._input_rect.centery)),
        )

        if active and self._input_focused and self._cursor_ms < 500:
            caret_x = (
                inner.left
                + self._input_font.size(self.config.model_path)[0]
                + 2
            )
            pygame.draw.line(
                self.surface,
                (220, 220, 230),
                (caret_x, self._input_rect.top + 6),
                (caret_x, self._input_rect.bottom - 6),
                1,
            )

    def _draw_start_button(self) -> None:
        enabled = self._start_enabled()
        if not enabled:
            bg = (55, 60, 72)
            border = (80, 85, 100)
            fg = (130, 134, 144)
        elif self._hovered_start:
            bg = (90, 185, 135)
            border = (140, 220, 180)
            fg = (10, 30, 20)
        else:
            bg = (70, 165, 115)
            border = (110, 200, 150)
            fg = (10, 30, 20)

        pygame.draw.rect(self.surface, bg, self._start_rect, border_radius=10)
        pygame.draw.rect(self.surface, border, self._start_rect, 2, border_radius=10)

        label = self._start_font.render("Start Game", True, fg)
        self.surface.blit(label, label.get_rect(center=self._start_rect.center))

    def _start_enabled(self) -> bool:
        if self.config.engine_kind == "model" and self.config.needs_engine:
            return bool(self.config.model_path.strip())
        return True
