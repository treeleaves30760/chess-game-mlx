"""Color themes and visual constants for the Chess GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Color type alias — (R, G, B) or (R, G, B, A)
# ---------------------------------------------------------------------------
Color = tuple[int, ...]


@dataclass(frozen=True)
class Theme:
    """Complete visual theme definition."""

    # Board squares
    light_square: Color = (240, 217, 181)
    dark_square: Color = (181, 136, 99)
    selected_square: Color = (255, 255, 0, 160)
    last_move_from: Color = (205, 210, 106, 160)
    last_move_to: Color = (170, 162, 58, 160)
    check_square: Color = (255, 80, 80, 200)
    legal_move_dot: Color = (0, 0, 0, 80)

    # Board border / coordinates
    board_border: Color = (100, 70, 50)
    coord_text: Color = (200, 200, 200)

    # Eval bar
    eval_bar_white: Color = (255, 255, 255)
    eval_bar_black: Color = (30, 30, 30)
    eval_bar_border: Color = (80, 80, 80)
    eval_bar_zero_line: Color = (180, 180, 180)

    # Move arrow colors (top-3, with alpha)
    arrow_color_1: Color = (0, 180, 80, 180)    # best move — green
    arrow_color_2: Color = (60, 120, 240, 150)  # 2nd — blue
    arrow_color_3: Color = (240, 160, 0, 130)   # 3rd — orange

    # Panels
    panel_bg: Color = (40, 44, 52)
    panel_border: Color = (70, 75, 85)
    panel_header: Color = (55, 60, 72)

    # Text
    text_primary: Color = (220, 220, 220)
    text_secondary: Color = (160, 160, 160)
    text_accent: Color = (100, 210, 120)
    text_warning: Color = (255, 200, 80)
    text_error: Color = (255, 100, 100)

    # Buttons
    btn_normal: Color = (60, 65, 80)
    btn_hover: Color = (80, 90, 110)
    btn_active: Color = (90, 130, 170)
    btn_text: Color = (220, 220, 220)

    # Mate banner (forced win/loss indicator)
    mate_banner_win: Color = (255, 200, 60)   # gold — we have mate
    mate_banner_loss: Color = (220, 80, 80)   # red  — we are getting mated

    # Window background
    window_bg: Color = (30, 34, 40)

    # Move list
    move_list_bg: Color = (35, 38, 46)
    move_highlight: Color = (70, 100, 150)
    move_number: Color = (140, 140, 160)
    move_white: Color = (220, 220, 220)
    move_black: Color = (180, 200, 220)


# Default theme instance
DEFAULT_THEME: Final[Theme] = Theme()


# ---------------------------------------------------------------------------
# Window / layout constants
# ---------------------------------------------------------------------------

WINDOW_W: Final[int] = 1200
WINDOW_H: Final[int] = 800
FPS: Final[int] = 60

# Layout zones (pixels)
EVAL_BAR_W: Final[int] = 30
BOARD_MARGIN: Final[int] = 4
RIGHT_PANEL_W: Final[int] = 280
BUTTON_BAR_H: Final[int] = 44
STATUS_BAR_H: Final[int] = 22
MOVE_LIST_H: Final[int] = 130

# Font sizes
FONT_LARGE: Final[int] = 22
FONT_MEDIUM: Final[int] = 16
FONT_SMALL: Final[int] = 13
FONT_TINY: Final[int] = 11

# Piece rendering
PIECE_FONT_SCALE: Final[float] = 0.80  # fraction of square size
SHOGI_PIECE_FONT_SCALE: Final[float] = 0.58  # kanji fit tighter in a square

# Arrow rendering
ARROW_SHAFT_WIDTH: Final[int] = 6
ARROW_HEAD_SIZE: Final[int] = 16

# Eval bar
EVAL_BAR_MAX_CP: Final[int] = 1000  # centipawns that fill the bar fully

# Menu card sizing
MENU_CARD_W: Final[int] = 180
MENU_CARD_H: Final[int] = 88
MENU_CARD_GAP: Final[int] = 18
MENU_SECTION_GAP: Final[int] = 32
