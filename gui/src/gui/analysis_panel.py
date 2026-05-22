"""Analysis panel renderer.

Renders into a pygame.Surface region:
- Eval bar (vertical, left side of board)
- Top-3 move list with scores
- PV line
- Depth / NPS / nodes stats
- Mate banner when the engine reports a forced win/loss.
"""

from __future__ import annotations

import math

import pygame

from gui.engine_client import InfoUpdate
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
        # Sticky mate snapshot: keeps the banner visible across short windows
        # where the engine flickers back to `score cp` because of a transient
        # |q| < 0.99 read (e.g. concurrent backprop race in the C++ MCTS).
        # Cleared only by reset() — i.e. on new game / undo / move played.
        self._sticky_mate: InfoUpdate | None = None

        self._fonts: dict[str, pygame.font.Font] = {}

    # -----------------------------------------------------------------------
    # State update
    # -----------------------------------------------------------------------

    def update_info(self, info: InfoUpdate) -> None:
        """Ingest a new InfoUpdate from the engine."""
        # A real search line always carries a PV. A PV-less update (e.g. a
        # mis-parsed "info string ..." line, or a malformed engine emit) has no
        # move to show and would only blank a real top-move slot if we stored
        # it. Drop it — defense in depth alongside the parser's info-string
        # guard so the #1 move never silently vanishes from the panel.
        if not info.pv:
            return
        # Replace entry for this multipv index
        idx = info.multipv - 1
        while len(self._infos) <= idx:
            self._infos.append(InfoUpdate())
        self._infos[idx] = info

        if info.multipv == 1:
            self._depth = info.depth
            self._nodes = info.nodes
            self._nps = info.nps
            # Latch onto a mate observation so the banner survives a brief
            # window of "score cp" lines from a racing MCTS reporter. We
            # keep the previous _score_cp (and stale eval bar) rather than
            # overwriting it with 0 from a `score mate N` line, which would
            # otherwise yank the eval bar to centre during the mate display.
            if self._is_mate_info(info):
                self._sticky_mate = info
            else:
                self._score_cp = info.score_cp

    def _is_mate_info(self, info: InfoUpdate) -> bool:
        """True if this info line represents a forced mate.

        Either explicit `score mate N` or the saturated `score cp` fallback
        for older engine builds (|cp| >= 29000).
        """
        return info.score_mate is not None or abs(info.score_cp) >= 29000

    def reset(self) -> None:
        """Clear all state (new game / undo / move played)."""
        self._infos = []
        self._score_cp = 0
        self._depth = 0
        self._nodes = 0
        self._nps = 0
        self._sticky_mate = None

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render(self) -> None:
        """Draw both the eval bar and the right analysis panel."""
        self._draw_eval_bar()
        self._draw_right_panel()

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

        mate = self._detect_mate()
        if mate is not None:
            # Peg the bar to whichever side is winning the mate, so the user
            # can't read a contradictory eval bar (centre/equal) while the
            # banner says "MATE in N".
            _, winning = mate
            fraction = 1.0 if winning else 0.0
            label_text = f"+M{mate[0]}" if winning else f"-M{mate[0]}"
        else:
            cp = max(-EVAL_BAR_MAX_CP, min(EVAL_BAR_MAX_CP, self._score_cp))
            # Map cp to [0.0, 1.0] — 0 cp → 0.5, positive → more white
            fraction = 0.5 + 0.5 * math.tanh(cp / 400.0)
            sign = "+" if self._score_cp >= 0 else ""
            label_text = f"{sign}{self._score_cp / 100:.2f}"
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
        label_surf = font.render(label_text, True, self.theme.text_primary)
        # Rotate 90 degrees and draw
        label_rot = pygame.transform.rotate(label_surf, 90)
        cx = r.left + r.width // 2
        cy = r.top + r.height // 2
        self.surface.blit(label_rot, label_rot.get_rect(center=(cx, cy)))

    # -----------------------------------------------------------------------
    # Right panel
    # -----------------------------------------------------------------------

    def _draw_right_panel(self) -> None:
        r = self.right_panel_rect
        # Panel background
        pygame.draw.rect(self.surface, self.theme.panel_bg, r)
        pygame.draw.rect(self.surface, self.theme.panel_border, r, 1)

        y = r.top + 8
        x_margin = r.left + 8
        max_w = r.width - 16

        # -- Mate banner (only when the top PV is a forced win/loss) --
        y = self._draw_mate_banner(x_margin, y, max_w)

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

    def _draw_eval_summary(self, x: int, y: int, max_w: int) -> int:
        """Draw eval score + depth/NPS/nodes. Returns new y."""
        header_font = self._font(FONT_MEDIUM, bold=True)
        small_font = self._font(FONT_SMALL)

        # Eval score — show "Mate ±N" form when a mate is locked in, so the
        # number agrees with the mate banner instead of falling to "+0.00"
        # whenever a `score mate N` line (cp=0) lands on top.
        mate = self._detect_mate()
        if mate is not None:
            n, winning = mate
            eval_str = f"Eval  {'+' if winning else '-'}M{n}"
            color = (self.theme.text_accent if winning
                     else self.theme.text_error)
        else:
            sign = "+" if self._score_cp >= 0 else ""
            eval_str = f"Eval  {sign}{self._score_cp / 100:.2f}"
            color = (self.theme.text_accent if self._score_cp >= 0
                     else self.theme.text_error)
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

    def _detect_mate(self) -> tuple[int, bool] | None:
        """If top PV is a forced mate, return (mate_in_N, white_winning). Else None.

        Engines may emit either ``score mate N`` (newer) or saturate ``score cp``
        near ±32000 (older). Both paths are handled here.

        Prefers the sticky snapshot when present so the banner stays visible
        across brief non-mate flickers from a racing MCTS reporter; the sticky
        snapshot is cleared in reset() when the position is known to change.
        """
        src = self._sticky_mate
        if src is None and self._infos:
            src = self._infos[0]
        if src is None:
            return None
        if src.score_mate is not None:
            n = abs(src.score_mate)
            return (max(1, n), src.score_mate > 0)
        if src.score_cp >= 29000:
            n = (32000 - src.score_cp) // 2 + 1
            return (max(1, n), True)
        if src.score_cp <= -29000:
            n = (32000 + src.score_cp) // 2 + 1
            return (max(1, n), False)
        return None

    def _draw_mate_banner(self, x: int, y: int, max_w: int) -> int:
        """Draw a gold/red mate banner when a forced mate is detected. Returns new y."""
        mate = self._detect_mate()
        if mate is None:
            return y
        n, winning = mate
        color = self.theme.mate_banner_win if winning else self.theme.mate_banner_loss
        label = f"⚑ MATE in {n} ⚑" if winning else f"⚑ Mated in {n} ⚑"

        font = self._font(FONT_MEDIUM, bold=True)
        text_surf = font.render(label, True, (0, 0, 0))
        h = text_surf.get_height() + 8
        rect = pygame.Rect(x, y, max_w, h)
        pygame.draw.rect(self.surface, color, rect, border_radius=4)
        pygame.draw.rect(self.surface, self.theme.panel_border, rect, 1, border_radius=4)
        self.surface.blit(text_surf, text_surf.get_rect(center=rect.center))
        return y + h + 6
