"""Analysis panel renderer.

Renders into a pygame.Surface region:
- Eval bar (vertical, left side of board)
- Top-3 move list with scores
- PV line
- Depth / NPS / nodes stats
- Ponder tree progress (during multi-ponder)
"""

from __future__ import annotations

import math

import pygame

from gui.engine_client import InfoUpdate
from gui.ponder_manager import PonderManager
from gui.themes import (
    DEFAULT_THEME,
    EVAL_BAR_MAX_CP,
    FONT_MEDIUM,
    FONT_SMALL,
    FONT_TINY,
    Theme,
)


class AnalysisPanel:
    """Draws the right-side analysis panel and left eval bar."""

    def __init__(
        self,
        surface: pygame.Surface,
        right_panel_rect: pygame.Rect,
        eval_bar_rect: pygame.Rect,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        self.surface = surface
        self.right_panel_rect = right_panel_rect
        self.eval_bar_rect = eval_bar_rect
        self.theme = theme

        # Current engine state
        self._infos: list[InfoUpdate] = []  # one per multipv, latest depth
        self._score_cp: int = 0
        self._depth: int = 0
        self._nodes: int = 0
        self._nps: int = 0

        self._fonts: dict[str, pygame.font.Font] = {}

    # -----------------------------------------------------------------------
    # State update
    # -----------------------------------------------------------------------

    def update_info(self, info: InfoUpdate) -> None:
        """Ingest a new InfoUpdate from the engine."""
        # Replace entry for this multipv index
        idx = info.multipv - 1
        while len(self._infos) <= idx:
            self._infos.append(InfoUpdate())
        self._infos[idx] = info

        if info.multipv == 1:
            self._score_cp = info.score_cp
            self._depth = info.depth
            self._nodes = info.nodes
            self._nps = info.nps

    def reset(self) -> None:
        """Clear all state (new game)."""
        self._infos = []
        self._score_cp = 0
        self._depth = 0
        self._nodes = 0
        self._nps = 0

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render(self, ponder_manager: PonderManager) -> None:
        """Draw both the eval bar and the right analysis panel."""
        self._draw_eval_bar()
        self._draw_right_panel(ponder_manager)

    def _get_font(self, name: str, size: int, bold: bool = False) -> pygame.font.Font:
        key = f"{name}_{size}_{bold}"
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont(name, size, bold=bold)
        return self._fonts[key]

    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        return self._get_font("sans", size, bold=bold)

    # -----------------------------------------------------------------------
    # Eval bar
    # -----------------------------------------------------------------------

    def _draw_eval_bar(self) -> None:
        r = self.eval_bar_rect
        # Background
        pygame.draw.rect(self.surface, self.theme.eval_bar_black, r)

        # White portion (from bottom, fraction based on score)
        cp = max(-EVAL_BAR_MAX_CP, min(EVAL_BAR_MAX_CP, self._score_cp))
        # Map cp to [0.0, 1.0] — 0 cp → 0.5, positive → more white
        fraction = 0.5 + 0.5 * math.tanh(cp / 400.0)
        white_h = int(r.height * fraction)
        white_rect = pygame.Rect(r.left, r.bottom - white_h, r.width, white_h)
        pygame.draw.rect(self.surface, self.theme.eval_bar_white, white_rect)

        # Zero line
        zero_y = r.top + r.height // 2
        pygame.draw.line(
            self.surface, self.theme.eval_bar_zero_line,
            (r.left, zero_y), (r.right, zero_y), 1
        )

        # Border
        pygame.draw.rect(self.surface, self.theme.eval_bar_border, r, 1)

        # Score label
        font = self._font(FONT_TINY)
        sign = "+" if self._score_cp >= 0 else ""
        label = f"{sign}{self._score_cp / 100:.2f}"
        label_surf = font.render(label, True, self.theme.text_primary)
        # Rotate 90 degrees and draw
        label_rot = pygame.transform.rotate(label_surf, 90)
        cx = r.left + r.width // 2
        cy = r.top + r.height // 2
        self.surface.blit(label_rot, label_rot.get_rect(center=(cx, cy)))

    # -----------------------------------------------------------------------
    # Right panel
    # -----------------------------------------------------------------------

    def _draw_right_panel(self, ponder_manager: PonderManager) -> None:
        r = self.right_panel_rect
        # Panel background
        pygame.draw.rect(self.surface, self.theme.panel_bg, r)
        pygame.draw.rect(self.surface, self.theme.panel_border, r, 1)

        y = r.top + 8
        x_margin = r.left + 8
        max_w = r.width - 16

        # -- Eval summary row --
        y = self._draw_eval_summary(x_margin, y, max_w)
        y += 4

        # -- Divider --
        pygame.draw.line(self.surface, self.theme.panel_border, (x_margin, y), (r.right - 8, y), 1)
        y += 8

        # -- Top-3 moves --
        y = self._draw_top_moves(x_margin, y, max_w)
        y += 4

        # -- PV line --
        y = self._draw_pv(x_margin, y, max_w)
        y += 4

        # -- Divider --
        pygame.draw.line(self.surface, self.theme.panel_border, (x_margin, y), (r.right - 8, y), 1)
        y += 8

        # -- Ponder trees --
        self._draw_ponder_trees(x_margin, y, max_w, ponder_manager)

    def _draw_eval_summary(self, x: int, y: int, max_w: int) -> int:
        """Draw eval score + depth/NPS/nodes. Returns new y."""
        header_font = self._font(FONT_MEDIUM, bold=True)
        small_font = self._font(FONT_SMALL)

        # Eval score
        sign = "+" if self._score_cp >= 0 else ""
        eval_str = f"Eval  {sign}{self._score_cp / 100:.2f}"
        color = self.theme.text_accent if self._score_cp >= 0 else self.theme.text_error
        eval_surf = header_font.render(eval_str, True, color)
        self.surface.blit(eval_surf, (x, y))
        y += eval_surf.get_height() + 4

        # Depth / NPS
        nps_k = self._nps // 1000
        depth_str = f"Depth {self._depth}  NPS {nps_k}K"
        depth_surf = small_font.render(depth_str, True, self.theme.text_secondary)
        self.surface.blit(depth_surf, (x, y))
        y += depth_surf.get_height() + 2

        # Nodes
        nodes_m = self._nodes / 1_000_000
        nodes_str = f"Nodes {nodes_m:.1f}M"
        nodes_surf = small_font.render(nodes_str, True, self.theme.text_secondary)
        self.surface.blit(nodes_surf, (x, y))
        y += nodes_surf.get_height() + 2

        return y

    def _draw_top_moves(self, x: int, y: int, max_w: int) -> int:
        """Draw top-3 moves with scores. Returns new y."""
        header_font = self._font(FONT_SMALL, bold=True)
        move_font = self._font(FONT_SMALL)

        header = header_font.render("Top-3 moves:", True, self.theme.text_primary)
        self.surface.blit(header, (x, y))
        y += header.get_height() + 2

        for i, info in enumerate(self._infos[:3]):
            if not info.pv:
                continue
            move_str = info.pv[0] if info.pv else "—"
            sign = "+" if info.score_cp >= 0 else ""
            line = f"  {i + 1}. {move_str:<8} {sign}{info.score_cp / 100:.2f}"
            color = self.theme.text_accent if i == 0 else self.theme.text_primary
            surf = move_font.render(line, True, color)
            self.surface.blit(surf, (x, y))
            y += surf.get_height() + 1

        return y

    def _draw_pv(self, x: int, y: int, max_w: int) -> int:
        """Draw the PV line for the best move. Returns new y."""
        small_font = self._font(FONT_SMALL)

        if not self._infos or not self._infos[0].pv:
            return y

        pv_moves = self._infos[0].pv[:8]
        pv_str = "PV: " + " ".join(pv_moves)

        # Word-wrap to max_w
        words = pv_str.split()
        lines: list[str] = []
        current = ""
        for word in words:
            test = current + (" " if current else "") + word
            if small_font.size(test)[0] > max_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        for line in lines[:2]:  # max 2 lines
            surf = small_font.render(line, True, self.theme.text_secondary)
            self.surface.blit(surf, (x, y))
            y += surf.get_height() + 1

        return y

    def _draw_ponder_trees(
        self,
        x: int,
        y: int,
        max_w: int,
        ponder_manager: PonderManager,
    ) -> int:
        """Draw ponder tree progress bars. Returns new y."""
        header_font = self._font(FONT_SMALL, bold=True)
        tree_font = self._font(FONT_SMALL)

        header_surf = header_font.render("Ponder trees:", True, self.theme.text_primary)
        self.surface.blit(header_surf, (x, y))
        y += header_surf.get_height() + 4

        trees = ponder_manager.trees_snapshot()
        if not trees:
            if ponder_manager.hit_bestmove:
                # Show hit info
                hit_color = self.theme.ponder_active
                hit_str = f"Hit! Best: {ponder_manager.hit_bestmove}"
                surf = tree_font.render(hit_str, True, hit_color)
                self.surface.blit(surf, (x, y))
                y += surf.get_height() + 2
            else:
                no_surf = tree_font.render("(not active)", True, self.theme.text_secondary)
                self.surface.blit(no_surf, (x, y))
                y += no_surf.get_height()
            return y

        bar_h = 8
        bar_gap = 4
        hit_tree = ponder_manager.hit_tree

        for tree in trees[:5]:
            is_hit = hit_tree is not None and tree.tree_id == hit_tree
            color = self.theme.ponder_active if is_hit else self.theme.ponder_inactive
            bullet = "●" if is_hit else "○"

            sign = "+" if tree.score_cp >= 0 else ""
            label = f"{bullet} {tree.opponent_move:<8} d{tree.depth:<3} {sign}{tree.score_cp / 100:.2f}"
            surf = tree_font.render(label, True, color)
            self.surface.blit(surf, (x, y))
            y += surf.get_height() + 1

            # Budget bar
            bar_w = int(max_w * tree.budget_weight)
            bar_rect = pygame.Rect(x, y, max_w, bar_h)
            fill_rect = pygame.Rect(x, y, bar_w, bar_h)
            pygame.draw.rect(self.surface, self.theme.ponder_bar_bg, bar_rect)
            bar_color = self.theme.ponder_active if is_hit else self.theme.ponder_bar_fill
            pygame.draw.rect(self.surface, bar_color, fill_rect)

            y += bar_h + bar_gap

        return y
