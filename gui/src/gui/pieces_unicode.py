"""Unicode piece glyph tables for chess and shogi rendering.

No image assets required — we draw text glyphs scaled to the square size.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Chess Unicode symbols
# Standard chess symbols from Unicode block U+2654–U+265F
# ---------------------------------------------------------------------------

CHESS_UNICODE: dict[str, str] = {
    # White pieces (uppercase letters in python-chess notation)
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    # Black pieces (lowercase)
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}

# Text shadow / outline colors for pieces on light vs dark squares
CHESS_PIECE_TEXT_COLOR: dict[str, tuple[int, int, int]] = {
    # White pieces on any square
    "K": (255, 255, 255),
    "Q": (255, 255, 255),
    "R": (255, 255, 255),
    "B": (255, 255, 255),
    "N": (255, 255, 255),
    "P": (255, 255, 255),
    # Black pieces
    "k": (20, 20, 20),
    "q": (20, 20, 20),
    "r": (20, 20, 20),
    "b": (20, 20, 20),
    "n": (20, 20, 20),
    "p": (20, 20, 20),
}

# Outline color (to ensure visibility on both square colors)
CHESS_PIECE_OUTLINE_COLOR: dict[str, tuple[int, int, int]] = {
    "K": (60, 60, 60),
    "Q": (60, 60, 60),
    "R": (60, 60, 60),
    "B": (60, 60, 60),
    "N": (60, 60, 60),
    "P": (60, 60, 60),
    "k": (200, 200, 200),
    "q": (200, 200, 200),
    "r": (200, 200, 200),
    "b": (200, 200, 200),
    "n": (200, 200, 200),
    "p": (200, 200, 200),
}

# ---------------------------------------------------------------------------
# Shogi CJK piece characters
# ---------------------------------------------------------------------------

# Normal pieces: (kanji, is_promoted)
SHOGI_UNICODE: dict[str, str] = {
    # Unpromoted
    "K": "王",   # King (older)
    "k": "玉",   # King (newer / second player variant)
    "R": "飛",   # Rook (hi-sha)
    "B": "角",   # Bishop (kaku-gyo)
    "G": "金",   # Gold General
    "S": "銀",   # Silver General
    "N": "桂",   # Knight (keima)
    "L": "香",   # Lance (kyosha)
    "P": "歩",   # Pawn (fu)
    # Promoted
    "+R": "龍",  # Promoted Rook (dragon)
    "+B": "馬",  # Promoted Bishop (horse)
    "+S": "全",  # Promoted Silver
    "+N": "圭",  # Promoted Knight
    "+L": "杏",  # Promoted Lance
    "+P": "と",  # Promoted Pawn (tokin)
}

SHOGI_GOTE_UNICODE: dict[str, str] = {
    # Gote (second player) pieces — often displayed upside down; we add suffix
    "K": "王",
    "k": "玉",
    "R": "飛",
    "B": "角",
    "G": "金",
    "S": "銀",
    "N": "桂",
    "L": "香",
    "P": "歩",
    "+R": "龍",
    "+B": "馬",
    "+S": "全",
    "+N": "圭",
    "+L": "杏",
    "+P": "と",
}

# Color for sente (first player) pieces
SHOGI_SENTE_COLOR: tuple[int, int, int] = (240, 230, 200)
# Color for gote (second player) pieces
SHOGI_GOTE_COLOR: tuple[int, int, int] = (80, 60, 40)
SHOGI_OUTLINE_COLOR: tuple[int, int, int] = (100, 80, 50)
