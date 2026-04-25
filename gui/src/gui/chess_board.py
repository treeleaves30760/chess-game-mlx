"""Chess board renderer (8x8) using python-chess for board state.

Renders:
- Squares with alternating light/dark colors
- Piece glyphs using Unicode characters
- Rank/file coordinate labels
- Highlights: selected square, last-move squares, check square, legal move dots
- Semi-transparent move arrows (drawn separately on an SRCALPHA overlay)
"""

from __future__ import annotations

import chess
import pygame

from gui.board_renderer import BoardRenderer
from gui.pieces_unicode import (
    CHESS_PIECE_OUTLINE_COLOR,
    CHESS_PIECE_TEXT_COLOR,
    CHESS_UNICODE,
)
from gui.themes import (
    DEFAULT_THEME,
    PIECE_FONT_SCALE,
    Theme,
)

# Fonts known to ship with chess Unicode glyphs (U+2654–U+265F), in preference
# order. `pygame.font.SysFont` silently falls back to the default font when no
# candidate is installed — and that default (e.g. Arial on macOS) may not
# include these glyphs, producing tofu boxes. Filter against get_fonts() so we
# only request names actually installed.
_CHESS_FONT_CANDIDATES: tuple[str, ...] = (
    "arialunicodems",    # macOS (Arial Unicode MS) — renders both chess + CJK
    "applesymbols",      # macOS (Apple Symbols) — clean outline chess glyphs
    "segoeuisymbol",     # Windows
    "dejavusans",        # Linux (common)
    "notosanssymbols2",  # Linux (Noto)
    "symbola",           # cross-platform if installed
    "unifont",           # fallback that covers the BMP
)


def _load_chess_piece_font(size: int) -> pygame.font.Font:
    """Load a font that can actually render chess Unicode glyphs."""
    available = set(pygame.font.get_fonts())
    for name in _CHESS_FONT_CANDIDATES:
        if name in available:
            return pygame.font.SysFont(name, size)
    return pygame.font.SysFont(None, size)


class MoveArrow:
    """Data class representing a ranked move arrow."""

    __slots__ = ("from_sq", "rank", "score_cp", "to_sq")

    def __init__(self, from_sq: int, to_sq: int, rank: int, score_cp: int) -> None:
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.rank = rank  # 1 = best, 2 = second, 3 = third
        self.score_cp = score_cp


class ChessBoardRenderer(BoardRenderer):
    """Renders an 8×8 chess board with python-chess Board state."""

    FILES = "abcdefgh"
    RANKS = "12345678"

    def __init__(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        board: chess.Board | None = None,
        theme: Theme = DEFAULT_THEME,
        flipped: bool = False,
    ) -> None:
        super().__init__(surface, rect, theme, flipped)
        self.board: chess.Board = board if board is not None else chess.Board()

        # UI state
        self.selected_square: int | None = None
        self.legal_targets: set[int] = set()
        self.last_move: chess.Move | None = None
        self.arrows: list[MoveArrow] = []

        # Fonts — initialized on first render
        self._piece_font: pygame.font.Font | None = None
        self._piece_font_size: int = 0  # size the cached piece font was built for
        self._coord_font: pygame.font.Font | None = None
        self._sq_size: int = 0

    # -----------------------------------------------------------------------
    # BoardRenderer interface
    # -----------------------------------------------------------------------

    def render(self) -> None:
        """Draw everything onto self.surface within self.rect."""
        sq = self._compute_sq_size()

        self._draw_squares(sq)
        self._draw_highlights(sq)
        self._draw_pieces(sq)
        self._draw_coordinates(sq)
        self._draw_arrows(sq)

    def square_at_pixel(self, px: int, py: int) -> int | None:
        """Return chess square index (0-63) for pixel (px, py), or None."""
        sq = self._compute_sq_size()
        bx = px - self.rect.left
        by = py - self.rect.top
        if not (0 <= bx < sq * 8 and 0 <= by < sq * 8):
            return None

        col = bx // sq
        row = by // sq

        if self.flipped:
            file_idx = 7 - col
            rank_idx = row
        else:
            file_idx = col
            rank_idx = 7 - row

        return chess.square(file_idx, rank_idx)

    def pixel_center_of_square(self, square: int) -> tuple[int, int]:
        """Return pixel center (x, y) of chess *square* in surface coordinates."""
        sq = self._compute_sq_size()
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        if self.flipped:
            col = 7 - file_idx
            row = rank_idx
        else:
            col = file_idx
            row = 7 - rank_idx

        cx = self.rect.left + col * sq + sq // 2
        cy = self.rect.top + row * sq + sq // 2
        return cx, cy

    # -----------------------------------------------------------------------
    # Public state setters
    # -----------------------------------------------------------------------

    def set_board(self, board: chess.Board) -> None:
        self.board = board

    def set_selected(self, square: int | None) -> None:
        self.selected_square = square
        if square is not None:
            self.legal_targets = {
                m.to_square for m in self.board.legal_moves if m.from_square == square
            }
        else:
            self.legal_targets = set()

    def set_last_move(self, move: chess.Move | None) -> None:
        self.last_move = move

    def set_arrows(self, arrows: list[MoveArrow]) -> None:
        self.arrows = arrows

    # -----------------------------------------------------------------------
    # Private drawing helpers
    # -----------------------------------------------------------------------

    def _compute_sq_size(self) -> int:
        """Compute square pixel size from available rect (8×8 board)."""
        sq = min(self.rect.width, self.rect.height) // 8
        self._sq_size = sq
        return sq

    def _get_piece_font(self, sq: int) -> pygame.font.Font:
        target_size = max(8, int(sq * PIECE_FONT_SCALE))
        if self._piece_font is None or self._piece_font_size != target_size:
            self._piece_font = _load_chess_piece_font(target_size)
            self._piece_font_size = target_size
        return self._piece_font

    def _get_coord_font(self) -> pygame.font.Font:
        if self._coord_font is None:
            self._coord_font = pygame.font.SysFont("sans", max(8, self._sq_size // 5))
        return self._coord_font

    def _square_rect(self, square: int, sq: int) -> pygame.Rect:
        """Return the pygame.Rect for a chess square."""
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        if self.flipped:
            col = 7 - file_idx
            row = rank_idx
        else:
            col = file_idx
            row = 7 - rank_idx

        x = self.rect.left + col * sq
        y = self.rect.top + row * sq
        return pygame.Rect(x, y, sq, sq)

    def _draw_squares(self, sq: int) -> None:
        for square in chess.SQUARES:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            is_light = (file_idx + rank_idx) % 2 == 1
            color = self.theme.light_square if is_light else self.theme.dark_square
            rect = self._square_rect(square, sq)
            pygame.draw.rect(self.surface, color, rect)

    def _draw_highlights(self, sq: int) -> None:
        # Use an SRCALPHA overlay for transparency
        overlay = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)

        # Last move squares
        if self.last_move is not None:
            for square in (self.last_move.from_square, self.last_move.to_square):
                r = self._square_rect(square, sq)
                r = r.move(-self.rect.left, -self.rect.top)
                pygame.draw.rect(overlay, self.theme.last_move_from, r)

        # Check highlight
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                r = self._square_rect(king_sq, sq)
                r = r.move(-self.rect.left, -self.rect.top)
                pygame.draw.rect(overlay, self.theme.check_square, r)

        # Selected square
        if self.selected_square is not None:
            r = self._square_rect(self.selected_square, sq)
            r = r.move(-self.rect.left, -self.rect.top)
            pygame.draw.rect(overlay, self.theme.selected_square, r)

        # Legal move dots
        dot_r = max(4, sq // 6)
        for target in self.legal_targets:
            cx, cy = self.pixel_center_of_square(target)
            cx -= self.rect.left
            cy -= self.rect.top
            pygame.draw.circle(overlay, self.theme.legal_move_dot, (cx, cy), dot_r)

        self.surface.blit(overlay, self.rect.topleft)

    def _draw_pieces(self, sq: int) -> None:
        font = self._get_piece_font(sq)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue

            symbol = piece.symbol()  # e.g. 'K', 'p'
            glyph = CHESS_UNICODE.get(symbol, "?")
            text_color = CHESS_PIECE_TEXT_COLOR.get(symbol, (180, 180, 180))
            outline_color = CHESS_PIECE_OUTLINE_COLOR.get(symbol, (0, 0, 0))

            rect = self._square_rect(square, sq)
            center = rect.center
            self.draw_centered_text(
                self.surface, font, glyph, center, text_color, outline_color
            )

    def _draw_coordinates(self, sq: int) -> None:
        font = self._get_coord_font()
        margin = max(2, sq // 8)
        for i in range(8):
            # Rank labels (left edge)
            rank_label = self.RANKS[i] if not self.flipped else self.RANKS[7 - i]
            row = 7 - i
            x = self.rect.left + margin
            y = self.rect.top + row * sq + margin
            rank_surf = font.render(rank_label, True, self.theme.coord_text)
            self.surface.blit(rank_surf, (x, y))

            # File labels (bottom edge)
            file_label = self.FILES[i] if not self.flipped else self.FILES[7 - i]
            col = i
            x = self.rect.left + col * sq + sq - rank_surf.get_width() - margin
            y = self.rect.top + 8 * sq - rank_surf.get_height() - margin
            file_surf = font.render(file_label, True, self.theme.coord_text)
            self.surface.blit(file_surf, (x, y))

    def _draw_arrows(self, sq: int) -> None:
        if not self.arrows:
            return

        arrow_colors = [
            self.theme.arrow_color_1,
            self.theme.arrow_color_2,
            self.theme.arrow_color_3,
        ]

        overlay = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)

        # Create a temporary renderer that draws into the overlay
        # (offset by -rect.topleft so coordinates align)
        temp_rect = pygame.Rect(0, 0, self.rect.width, self.rect.height)
        temp_renderer = _OffsetChessRenderer(
            surface=overlay,
            rect=temp_rect,
            board=self.board,
            theme=self.theme,
            flipped=self.flipped,
            offset=self.rect.topleft,
            sq_size=sq,
        )

        label_font = pygame.font.SysFont(None, max(14, sq // 4), bold=True)

        for arrow in sorted(self.arrows, key=lambda a: -a.rank):  # draw best last (on top)
            color_idx = min(arrow.rank - 1, len(arrow_colors) - 1)
            color = arrow_colors[color_idx]
            temp_renderer.draw_arrow(overlay, arrow.from_sq, arrow.to_sq, color)
            self._draw_arrow_label(
                overlay, temp_renderer, arrow, color, label_font, sq
            )

        self.surface.blit(overlay, self.rect.topleft)

    @staticmethod
    def _format_score_cp(cp: int) -> str:
        """Format centipawns as a short signed decimal string (+0.35 / -1.20 / #M5)."""
        if cp >= 29000:    # mate-in-N (our engine emits ~32000 for mate)
            return f"#M{(32000 - cp) // 2 + 1}"
        if cp <= -29000:
            return f"#-M{(32000 + cp) // 2 + 1}"
        return f"{cp / 100:+.2f}"

    # Badge corner anchor by rank: each rank occupies a different corner of
    # the destination square so when multiple arrows share a target square
    # (e.g. top-1 and top-2 both pointing to e4) the badges don't stack on
    # top of each other and obscure each other. Rank 4+ (rare) reuses the
    # bottom-left corner.
    _BADGE_CORNERS: tuple[str, ...] = ("TL", "TR", "BR", "BL")

    def _draw_arrow_label(
        self,
        overlay: pygame.Surface,
        temp_renderer: "_OffsetChessRenderer",
        arrow: "MoveArrow",
        color: tuple[int, int, int, int],
        font: pygame.font.Font,
        sq: int,
    ) -> None:
        """Draw a '#rank  +cp' badge at the arrow's destination square."""
        cx, cy = temp_renderer.pixel_center_of_square(arrow.to_sq)
        text = f"#{arrow.rank}  {self._format_score_cp(arrow.score_cp)}"
        text_surf = font.render(text, True, (255, 255, 255))
        tw, th = text_surf.get_size()

        pad_x, pad_y = 4, 2
        bg_w = tw + pad_x * 2
        bg_h = th + pad_y * 2
        margin = 2

        # Pick a corner of the destination square based on rank so different
        # arrows targeting the same square don't collide.
        corner = self._BADGE_CORNERS[(arrow.rank - 1) % len(self._BADGE_CORNERS)]
        if corner == "TL":
            bx = cx - sq // 2 + margin
            by = cy - sq // 2 + margin
        elif corner == "TR":
            bx = cx + sq // 2 - margin - bg_w
            by = cy - sq // 2 + margin
        elif corner == "BR":
            bx = cx + sq // 2 - margin - bg_w
            by = cy + sq // 2 - margin - bg_h
        else:  # BL
            bx = cx - sq // 2 + margin
            by = cy + sq // 2 - margin - bg_h

        # Clamp to overlay bounds in case the destination is at a board edge.
        bx = max(0, min(bx, overlay.get_width() - bg_w))
        by = max(0, min(by, overlay.get_height() - bg_h))

        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg.fill((color[0], color[1], color[2], 235))
        pygame.draw.rect(bg, (0, 0, 0, 235), bg.get_rect(), 1)
        overlay.blit(bg, (bx, by))
        overlay.blit(text_surf, (bx + pad_x, by + pad_y))


class _OffsetChessRenderer(BoardRenderer):
    """Helper renderer for drawing arrows into an offset overlay surface."""

    def __init__(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        board: chess.Board,
        theme: Theme,
        flipped: bool,
        offset: tuple[int, int],
        sq_size: int,
    ) -> None:
        super().__init__(surface, rect, theme, flipped)
        self.board = board
        self._offset = offset
        self._sq = sq_size

    def render(self) -> None:
        pass  # Not used directly

    def square_at_pixel(self, px: int, py: int) -> int | None:
        return None

    def pixel_center_of_square(self, square: int) -> tuple[int, int]:
        sq = self._sq
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        if self.flipped:
            col = 7 - file_idx
            row = rank_idx
        else:
            col = file_idx
            row = 7 - rank_idx

        # Draw into overlay (no offset needed, overlay is positioned at rect.topleft)
        cx = col * sq + sq // 2
        cy = row * sq + sq // 2
        return cx, cy
