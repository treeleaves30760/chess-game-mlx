"""Shogi (将棋) board renderer backed by python-shogi.

Renders a full 9×9 shogi board plus hand-piece trays (駒台) for both players,
with correct piece kanji, gote pieces rotated 180°, drop-selection targets, and
an inline promotion prompt when the user attempts to enter the promotion zone.

Coordinate conventions (python-shogi):
- `square_index = rank_index * 9 + file_index`
- `rank_index`: 0 = rank A (top — gote back rank), 8 = rank I (bottom — sente)
- `file_index`: 0 = file 9 (leftmost from sente), 8 = file 1 (rightmost)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pygame
import shogi

from gui.board_renderer import BoardRenderer
from gui.themes import (
    DEFAULT_THEME,
    FONT_MEDIUM,
    FONT_SMALL,
    FONT_TINY,
    SHOGI_PIECE_FONT_SCALE,
    Theme,
)

# Kanji for each piece type. Gote pieces share the same glyph but are rotated.
_PIECE_KANJI: dict[int, str] = {
    shogi.PAWN: "歩",
    shogi.LANCE: "香",
    shogi.KNIGHT: "桂",
    shogi.SILVER: "銀",
    shogi.GOLD: "金",
    shogi.BISHOP: "角",
    shogi.ROOK: "飛",
    shogi.KING: "玉",      # gote king conventionally uses 玉; sente uses 王.
    shogi.PROM_PAWN: "と",
    shogi.PROM_LANCE: "杏",
    shogi.PROM_KNIGHT: "圭",
    shogi.PROM_SILVER: "全",
    shogi.PROM_BISHOP: "馬",
    shogi.PROM_ROOK: "龍",
}

# Drop-eligible unpromoted piece types, in the order shown on the駒台 tray.
_HAND_ORDER: tuple[int, ...] = (
    shogi.ROOK,
    shogi.BISHOP,
    shogi.GOLD,
    shogi.SILVER,
    shogi.KNIGHT,
    shogi.LANCE,
    shogi.PAWN,
)

# Fonts that support Japanese kanji on the target platform. Order matters:
# Hiragino Sans GB and Arial Unicode MS ship with macOS and render both chess
# Unicode and CJK cleanly; the Linux/Windows names are fallbacks.
_KANJI_FONT_CANDIDATES: tuple[str, ...] = (
    "arialunicodems",
    "hiraginosansgb",
    "hiraginosans",
    "notosanscjkjp",
    "notosanscjksc",
    "yugothic",
    "msgothic",
)


def _load_kanji_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Load a CJK-capable font, verifying the resolved path actually exists.

    `pygame.font.match_font` can point to ghost fonts from stale caches (e.g.
    /opt/X11 entries that no longer exist), so we must probe each candidate.
    """
    available = set(pygame.font.get_fonts())
    for name in _KANJI_FONT_CANDIDATES:
        if name not in available:
            continue
        path = pygame.font.match_font(name, bold=bold)
        if path and os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except OSError:
                continue
    return pygame.font.SysFont("sans", size, bold=bold)


@dataclass
class _PendingPromotion:
    """A move that reaches the promotion zone and needs the user's decision."""

    move_promote: shogi.Move   # the promoted version
    move_plain: shogi.Move     # the unpromoted version (None-equivalent if forced)
    forced: bool               # True when only the promoted move is legal


class ShogiBoardRenderer(BoardRenderer):
    """Renders a 9×9 shogi board + hand trays with click-driven interaction."""

    # Rank labels appear on the left edge (a-i top to bottom from black's view)
    RANK_LABELS = "abcdefghi"
    # File labels appear on the top (9..1 left to right from black's view)
    FILE_LABELS = "987654321"

    def __init__(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        board: shogi.Board | None = None,
        theme: Theme = DEFAULT_THEME,
        flipped: bool = False,
    ) -> None:
        super().__init__(surface, rect, theme, flipped)
        self.board: shogi.Board = board if board is not None else shogi.Board()

        # UI state
        self.selected_square: int | None = None
        self.selected_hand: tuple[int, int] | None = None  # (color, piece_type)
        self.legal_targets: set[int] = set()
        self.last_move: shogi.Move | None = None
        self.pending_promotion: _PendingPromotion | None = None

        # Geometry — recomputed each render
        self._sq: int = 0
        self._grid_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._sente_tray: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._gote_tray: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._hand_rects: dict[tuple[int, int], pygame.Rect] = {}
        self._promo_buttons: dict[str, pygame.Rect] = {}

        # Fonts
        self._piece_font: pygame.font.Font | None = None
        self._piece_font_size: int = 0
        self._coord_font = pygame.font.SysFont("sans", FONT_TINY)
        # Tray header + button labels include 先/後/成 — need CJK support.
        self._hand_label_font = _load_kanji_font(FONT_SMALL, bold=True)
        self._promo_font = _load_kanji_font(FONT_MEDIUM, bold=True)

    # -----------------------------------------------------------------------
    # BoardRenderer interface
    # -----------------------------------------------------------------------

    def render(self) -> None:
        self._compute_geometry()
        self._draw_background()
        self._draw_trays()
        self._draw_grid()
        self._draw_highlights()
        self._draw_pieces()
        self._draw_hand_pieces()
        self._draw_coordinates()
        if self.pending_promotion is not None:
            self._draw_promotion_prompt()

    def square_at_pixel(self, px: int, py: int) -> int | None:
        """Return board square 0..80 for pixel (px, py); None if outside grid."""
        if self._sq == 0:
            self._compute_geometry()
        if not self._grid_rect.collidepoint(px, py):
            return None
        bx = px - self._grid_rect.left
        by = py - self._grid_rect.top
        col = bx // self._sq
        row = by // self._sq
        if not (0 <= col < 9 and 0 <= row < 9):
            return None
        # Non-flipped: black at bottom, files 9..1 left->right, ranks a..i top->bottom
        # So col 0 = file index 0 (file "9"), row 0 = rank index 0 (rank "a")
        if self.flipped:
            file_idx = 8 - int(col)
            rank_idx = 8 - int(row)
        else:
            file_idx = int(col)
            rank_idx = int(row)
        return rank_idx * 9 + file_idx

    def hand_piece_at_pixel(self, px: int, py: int) -> tuple[int, int] | None:
        """Return (color, piece_type) if (px, py) hit a hand piece stack."""
        for key, rect in self._hand_rects.items():
            if rect.collidepoint(px, py):
                return key
        return None

    def promo_button_at_pixel(self, px: int, py: int) -> str | None:
        """Return 'promote' / 'plain' if the user clicked a promotion-prompt button."""
        for name, rect in self._promo_buttons.items():
            if rect.collidepoint(px, py):
                return name
        return None

    def pixel_center_of_square(self, square: int) -> tuple[int, int]:
        if self._sq == 0:
            self._compute_geometry()
        file_idx = square % 9
        rank_idx = square // 9
        if self.flipped:
            col = 8 - file_idx
            row = 8 - rank_idx
        else:
            col = file_idx
            row = rank_idx
        cx = self._grid_rect.left + col * self._sq + self._sq // 2
        cy = self._grid_rect.top + row * self._sq + self._sq // 2
        return cx, cy

    # -----------------------------------------------------------------------
    # Public state setters
    # -----------------------------------------------------------------------

    def set_board(self, board: shogi.Board) -> None:
        self.board = board

    def set_selected_square(self, square: int | None) -> None:
        self.selected_square = square
        self.selected_hand = None
        self.legal_targets = set()
        if square is None:
            return
        for mv in self.board.legal_moves:
            if mv.from_square == square and mv.drop_piece_type is None:
                self.legal_targets.add(mv.to_square)

    def set_selected_hand(self, color: int, piece_type: int) -> None:
        # Can only use hand pieces on the side to move.
        self.selected_square = None
        if self.board.turn != color:
            self.selected_hand = None
            self.legal_targets = set()
            return
        self.selected_hand = (color, piece_type)
        self.legal_targets = set()
        for mv in self.board.legal_moves:
            if mv.drop_piece_type == piece_type:
                self.legal_targets.add(mv.to_square)

    def clear_selection(self) -> None:
        self.selected_square = None
        self.selected_hand = None
        self.legal_targets = set()

    def set_last_move(self, move: shogi.Move | None) -> None:
        self.last_move = move

    # -----------------------------------------------------------------------
    # Promotion helpers
    # -----------------------------------------------------------------------

    def build_candidate_move(
        self, from_square: int, to_square: int
    ) -> shogi.Move | _PendingPromotion | None:
        """Decide whether the user's click implies an immediate move or a prompt.

        Returns:
        - `shogi.Move` when a single legal move matches (no prompt needed).
        - `_PendingPromotion` when both promoted and non-promoted variants are
          legal (caller should ask the user which to pick).
        - `None` when neither variant is legal.
        """
        candidates = [
            mv
            for mv in self.board.legal_moves
            if mv.from_square == from_square
            and mv.to_square == to_square
            and mv.drop_piece_type is None
        ]
        if not candidates:
            return None

        promoted = next((m for m in candidates if m.promotion), None)
        plain = next((m for m in candidates if not m.promotion), None)

        if promoted and plain:
            return _PendingPromotion(move_promote=promoted, move_plain=plain, forced=False)
        if promoted and not plain:
            # Forced promotion (e.g. pawn reaching last rank). Still ask for
            # symmetry, but with only the Promote button enabled.
            return _PendingPromotion(
                move_promote=promoted, move_plain=promoted, forced=True
            )
        return plain

    def build_candidate_drop(
        self, piece_type: int, to_square: int
    ) -> shogi.Move | None:
        """Look up the legal drop move (if any) for *piece_type* onto *to_square*."""
        for mv in self.board.legal_moves:
            if (
                mv.drop_piece_type == piece_type
                and mv.to_square == to_square
                and mv.from_square is None
            ):
                return mv
        return None

    def set_pending_promotion(self, pending: _PendingPromotion | None) -> None:
        self.pending_promotion = pending

    # -----------------------------------------------------------------------
    # Geometry / layout
    # -----------------------------------------------------------------------

    def _compute_geometry(self) -> None:
        """Lay out hand trays at top/bottom with a centered 9x9 grid between."""
        r = self.rect
        tray_h = max(56, r.height // 11)
        # Provide breathing room around the grid.
        avail_h = r.height - 2 * tray_h - 12
        avail_w = r.width - 12
        board_side = min(avail_w, avail_h)
        # Snap to multiple of 9 so squares are integer-sized.
        sq = max(24, board_side // 9)
        board_side = sq * 9

        grid_x = r.left + (r.width - board_side) // 2
        grid_y = r.top + tray_h + 6 + (avail_h - board_side) // 2

        self._sq = sq
        self._grid_rect = pygame.Rect(grid_x, grid_y, board_side, board_side)

        tray_w = board_side + 16
        tray_x = r.left + (r.width - tray_w) // 2
        gote_tray_y = r.top + 2
        sente_tray_y = self._grid_rect.bottom + 6

        if self.flipped:
            self._sente_tray = pygame.Rect(tray_x, gote_tray_y, tray_w, tray_h)
            self._gote_tray = pygame.Rect(tray_x, sente_tray_y, tray_w, tray_h)
        else:
            self._gote_tray = pygame.Rect(tray_x, gote_tray_y, tray_w, tray_h)
            self._sente_tray = pygame.Rect(tray_x, sente_tray_y, tray_w, tray_h)

        self._hand_rects = {}
        self._promo_buttons = {}

    def _get_piece_font(self) -> pygame.font.Font:
        size = max(12, int(self._sq * SHOGI_PIECE_FONT_SCALE))
        if self._piece_font is None or self._piece_font_size != size:
            self._piece_font = _load_kanji_font(size, bold=True)
            self._piece_font_size = size
        return self._piece_font

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def _draw_background(self) -> None:
        pygame.draw.rect(self.surface, (40, 36, 28), self.rect)

    def _draw_trays(self) -> None:
        """Draw the two hand trays (駒台) with color-coded labels."""
        for color, rect, label in (
            (shogi.BLACK, self._sente_tray, "先 SENTE"),
            (shogi.WHITE, self._gote_tray, "後 GOTE"),
        ):
            pygame.draw.rect(self.surface, (55, 48, 35), rect, border_radius=4)
            pygame.draw.rect(self.surface, (100, 85, 60), rect, 1, border_radius=4)
            lbl_color = (230, 210, 140) if color == shogi.BLACK else (200, 220, 240)
            lbl = self._hand_label_font.render(label, True, lbl_color)
            self.surface.blit(
                lbl,
                lbl.get_rect(midleft=(rect.left + 8, rect.top + 12)),
            )

    def _draw_grid(self) -> None:
        r = self._grid_rect
        # Tatami-coloured panel
        pygame.draw.rect(self.surface, (235, 210, 150), r)
        # Grid lines
        for i in range(10):
            pygame.draw.line(
                self.surface, (60, 45, 20),
                (r.left, r.top + i * self._sq),
                (r.right, r.top + i * self._sq),
                1,
            )
            pygame.draw.line(
                self.surface, (60, 45, 20),
                (r.left + i * self._sq, r.top),
                (r.left + i * self._sq, r.bottom),
                1,
            )
        # Thicker border
        pygame.draw.rect(self.surface, (60, 45, 20), r, 2)
        # Traditional star markers (星) between 3rd-6th ranks/files.
        for sq in (3 * 9 + 3, 3 * 9 + 6, 6 * 9 + 3, 6 * 9 + 6):
            cx, cy = self._pixel_corner_of_square(sq)
            pygame.draw.circle(self.surface, (60, 45, 20), (cx, cy), 3)

    def _pixel_corner_of_square(self, square: int) -> tuple[int, int]:
        """Bottom-right corner (in grid coords) — used for star markers."""
        file_idx = square % 9
        rank_idx = square // 9
        if self.flipped:
            col = 8 - file_idx
            row = 8 - rank_idx
        else:
            col = file_idx
            row = rank_idx
        return (
            self._grid_rect.left + (col + 1) * self._sq,
            self._grid_rect.top + (row + 1) * self._sq,
        )

    def _square_rect(self, square: int) -> pygame.Rect:
        cx, cy = self.pixel_center_of_square(square)
        s = self._sq
        return pygame.Rect(cx - s // 2, cy - s // 2, s, s)

    def _draw_highlights(self) -> None:
        overlay = pygame.Surface(
            (self._grid_rect.width, self._grid_rect.height), pygame.SRCALPHA
        )
        # Last move
        if self.last_move is not None:
            for sq in (self.last_move.from_square, self.last_move.to_square):
                if sq is None:
                    continue
                r = self._square_rect(sq).move(
                    -self._grid_rect.left, -self._grid_rect.top
                )
                pygame.draw.rect(overlay, (205, 210, 106, 120), r)

        # Check highlight
        if self.board.is_check():
            king_sq = self.board.king_squares[self.board.turn]
            if king_sq is not None:
                r = self._square_rect(king_sq).move(
                    -self._grid_rect.left, -self._grid_rect.top
                )
                pygame.draw.rect(overlay, (255, 90, 90, 150), r)

        # Selected square
        if self.selected_square is not None:
            r = self._square_rect(self.selected_square).move(
                -self._grid_rect.left, -self._grid_rect.top
            )
            pygame.draw.rect(overlay, (255, 230, 100, 140), r)

        # Legal targets (dots for normal moves, rings for drops)
        dot_r = max(4, self._sq // 6)
        for sq in self.legal_targets:
            cx, cy = self.pixel_center_of_square(sq)
            cx -= self._grid_rect.left
            cy -= self._grid_rect.top
            if self.selected_hand is not None:
                pygame.draw.circle(overlay, (40, 160, 240, 150), (cx, cy), dot_r, 3)
            else:
                pygame.draw.circle(overlay, (30, 30, 30, 110), (cx, cy), dot_r)

        self.surface.blit(overlay, self._grid_rect.topleft)

    def _draw_pieces(self) -> None:
        font = self._get_piece_font()
        for sq in range(81):
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            kanji = _piece_kanji_for(piece)
            surf = self._render_piece_glyph(font, kanji, piece.color)
            cx, cy = self.pixel_center_of_square(sq)
            self.surface.blit(surf, surf.get_rect(center=(cx, cy)))

    def _render_piece_glyph(
        self, font: pygame.font.Font, kanji: str, color: int
    ) -> pygame.Surface:
        text_color = (20, 20, 20) if color == shogi.BLACK else (30, 30, 30)
        surf = font.render(kanji, True, text_color)
        if color == shogi.WHITE:
            surf = pygame.transform.rotate(surf, 180)
        return surf

    def _draw_hand_pieces(self) -> None:
        """Draw each player's captured pieces in their tray."""
        font = self._get_piece_font()
        for color in (shogi.BLACK, shogi.WHITE):
            tray = self._sente_tray if color == shogi.BLACK else self._gote_tray
            counter = self.board.pieces_in_hand[color]
            # Position pieces in a row after the label.
            piece_w = tray.height - 12
            x = tray.left + 90
            y = tray.top + 6
            for ptype in _HAND_ORDER:
                count = counter.get(ptype, 0)
                if count <= 0:
                    continue
                rect = pygame.Rect(x, y, piece_w, tray.height - 12)
                self._hand_rects[(color, ptype)] = rect
                is_selected = self.selected_hand == (color, ptype)

                bg = (90, 75, 50) if is_selected else (70, 58, 38)
                border = (220, 200, 140) if is_selected else (120, 100, 70)
                pygame.draw.rect(self.surface, bg, rect, border_radius=4)
                pygame.draw.rect(self.surface, border, rect, 2 if is_selected else 1, border_radius=4)

                kanji = _PIECE_KANJI[ptype]
                glyph = self._render_piece_glyph(font, kanji, color)
                # Scale to fit tray height
                max_side = rect.height - 4
                if glyph.get_height() > max_side:
                    factor = max_side / glyph.get_height()
                    glyph = pygame.transform.smoothscale(
                        glyph,
                        (int(glyph.get_width() * factor), max_side),
                    )
                self.surface.blit(
                    glyph,
                    glyph.get_rect(center=(rect.centerx, rect.centery)),
                )

                if count > 1:
                    count_surf = self._hand_label_font.render(
                        f"×{count}", True, (240, 220, 160)
                    )
                    self.surface.blit(
                        count_surf,
                        count_surf.get_rect(bottomright=(rect.right - 2, rect.bottom - 2)),
                    )
                x += piece_w + 6

    def _draw_coordinates(self) -> None:
        r = self._grid_rect
        margin = 4
        label_color = (210, 190, 150)
        for i in range(9):
            file_lbl = self.FILE_LABELS[i] if not self.flipped else self.FILE_LABELS[8 - i]
            rank_lbl = self.RANK_LABELS[i] if not self.flipped else self.RANK_LABELS[8 - i]

            file_surf = self._coord_font.render(file_lbl, True, label_color)
            fx = r.left + i * self._sq + self._sq // 2 - file_surf.get_width() // 2
            self.surface.blit(file_surf, (fx, r.top - file_surf.get_height() - 2))

            rank_surf = self._coord_font.render(rank_lbl, True, label_color)
            ry = r.top + i * self._sq + self._sq // 2 - rank_surf.get_height() // 2
            self.surface.blit(rank_surf, (r.left - rank_surf.get_width() - margin, ry))

    def _draw_promotion_prompt(self) -> None:
        assert self.pending_promotion is not None
        pending = self.pending_promotion

        # Center a modal panel near the destination square
        to_sq = pending.move_promote.to_square
        cx, cy = self.pixel_center_of_square(to_sq)
        panel_w = max(220, self._sq * 4)
        panel_h = 90
        panel_rect = pygame.Rect(
            cx - panel_w // 2,
            cy - panel_h // 2,
            panel_w,
            panel_h,
        ).clamp(self.rect)

        pygame.draw.rect(self.surface, (30, 30, 40), panel_rect, border_radius=8)
        pygame.draw.rect(self.surface, (180, 180, 200), panel_rect, 2, border_radius=8)

        title = self._promo_font.render(
            "成りますか？  Promote?", True, (240, 240, 240)
        )
        self.surface.blit(
            title, title.get_rect(midtop=(panel_rect.centerx, panel_rect.top + 8))
        )

        # Two buttons
        btn_y = panel_rect.top + panel_rect.height - 32
        btn_w = 90
        btn_h = 26
        promote_rect = pygame.Rect(
            panel_rect.centerx - btn_w - 6, btn_y, btn_w, btn_h
        )
        plain_rect = pygame.Rect(panel_rect.centerx + 6, btn_y, btn_w, btn_h)
        self._promo_buttons = {"promote": promote_rect, "plain": plain_rect}

        # Promote (Yes) button — always enabled
        pygame.draw.rect(self.surface, (70, 165, 115), promote_rect, border_radius=4)
        yes_label = self._hand_label_font.render("Yes (成)", True, (15, 25, 20))
        self.surface.blit(yes_label, yes_label.get_rect(center=promote_rect.center))

        # Plain (No) — greyed out when promotion is forced
        plain_bg = (60, 65, 80) if not pending.forced else (40, 42, 50)
        plain_fg = (230, 230, 230) if not pending.forced else (120, 120, 130)
        pygame.draw.rect(self.surface, plain_bg, plain_rect, border_radius=4)
        no_label = self._hand_label_font.render("No (不成)", True, plain_fg)
        self.surface.blit(no_label, no_label.get_rect(center=plain_rect.center))

        if pending.forced:
            # Only promote is actually selectable
            self._promo_buttons.pop("plain", None)


def _piece_kanji_for(piece: shogi.Piece) -> str:
    """Pick the correct kanji, using 王 for sente king vs 玉 for gote king."""
    if piece.piece_type == shogi.KING and piece.color == shogi.BLACK:
        return "王"
    return _PIECE_KANJI[piece.piece_type]


# Re-export pending promotion type so callers (controller / app) can reference it.
PendingPromotion = _PendingPromotion
