"""Abstract base class for board renderers.

Both ChessBoardRenderer and ShogiBoardRenderer inherit from this.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pygame

from gui.themes import ARROW_HEAD_SIZE, ARROW_SHAFT_WIDTH, DEFAULT_THEME, Theme

if TYPE_CHECKING:
    pass


class BoardRenderer(ABC):
    """Abstract renderer — subclasses implement game-specific drawing."""

    def __init__(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        theme: Theme = DEFAULT_THEME,
        flipped: bool = False,
    ) -> None:
        self.surface = surface
        self.rect = rect
        self.theme = theme
        self.flipped = flipped

    @abstractmethod
    def render(self) -> None:
        """Draw the full board into self.surface at self.rect."""
        ...

    @abstractmethod
    def square_at_pixel(self, px: int, py: int) -> int | None:
        """Return the logical square index at pixel (px, py), or None if outside board."""
        ...

    @abstractmethod
    def pixel_center_of_square(self, square: int) -> tuple[int, int]:
        """Return pixel center (x, y) of a logical square."""
        ...

    # -----------------------------------------------------------------------
    # Shared arrow drawing utility
    # -----------------------------------------------------------------------

    def draw_arrow(
        self,
        surface: pygame.Surface,
        from_sq: int,
        to_sq: int,
        color: tuple[int, ...],
        shaft_width: int = ARROW_SHAFT_WIDTH,
        head_size: int = ARROW_HEAD_SIZE,
    ) -> None:
        """Draw a semi-transparent arrow from *from_sq* to *to_sq* on *surface*."""
        fx, fy = self.pixel_center_of_square(from_sq)
        tx, ty = self.pixel_center_of_square(to_sq)

        # Compute direction vector
        dx = tx - fx
        dy = ty - fy
        length = math.hypot(dx, dy)
        if length < 1:
            return

        ux = dx / length
        uy = dy / length

        # Arrow shaft end (pulled back by head_size so head doesn't overshoot)
        shaft_end_x = tx - ux * head_size
        shaft_end_y = ty - uy * head_size

        # Perpendicular vector
        px_v = -uy * shaft_width / 2
        py_v = ux * shaft_width / 2

        # Shaft as a polygon
        shaft_pts = [
            (int(fx + px_v), int(fy + py_v)),
            (int(fx - px_v), int(fy - py_v)),
            (int(shaft_end_x - px_v), int(shaft_end_y - py_v)),
            (int(shaft_end_x + px_v), int(shaft_end_y + py_v)),
        ]

        # Arrow head
        perp_x = -uy * head_size * 0.6
        perp_y = ux * head_size * 0.6
        head_pts = [
            (int(tx), int(ty)),
            (int(shaft_end_x + perp_x), int(shaft_end_y + perp_y)),
            (int(shaft_end_x - perp_x), int(shaft_end_y - perp_y)),
        ]

        rgba = tuple(color) if len(color) == 4 else (*color, 200)
        pygame.draw.polygon(surface, rgba, shaft_pts)
        pygame.draw.polygon(surface, rgba, head_pts)

    # -----------------------------------------------------------------------
    # Helper to draw text centered in a rect
    # -----------------------------------------------------------------------

    @staticmethod
    def draw_centered_text(
        surface: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        center: tuple[int, int],
        color: tuple[int, ...],
        outline_color: tuple[int, ...] | None = None,
    ) -> None:
        """Render text centered at *center*, with optional 1-px outline."""
        if outline_color is not None:
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                outline_surf = font.render(text, True, outline_color)
                outline_rect = outline_surf.get_rect(center=(center[0] + ox, center[1] + oy))
                surface.blit(outline_surf, outline_rect)

        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center)
        surface.blit(text_surf, text_rect)
