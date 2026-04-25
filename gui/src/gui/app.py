"""Chess + Shogi AI GUI — Pygame main loop with a home-screen menu.

Entry points:
    uv run chess-mlx-gui [OPTIONS]
    uv run python -m gui.app [OPTIONS]

The app launches into the home-screen menu where the user picks the game
(chess / shogi), engine (rule-based or neural-model path), play mode
(AI vs AI / Human vs AI / Human vs Human), and — for Human vs AI — which side
to play. Clicking Start transitions to the game scene with that configuration;
clicking Menu in the game bar (or pressing Esc) returns to the menu.

CLI flags pre-fill the menu selections but don't skip it — the menu always
lets the user adjust before starting.
"""

from __future__ import annotations

import os
import time

import click
import pygame

from gui.game_scene import GameScene, GameSceneAction
from gui.menu import GameConfig, MenuAction, MenuScene
from gui.themes import FPS, WINDOW_H, WINDOW_W


class ChessApp:
    """Top-level Pygame application with menu ↔ game scene switching."""

    def __init__(
        self,
        initial_config: GameConfig | None = None,
        headless: bool = False,
    ) -> None:
        self._initial_config = initial_config or GameConfig()
        self._headless = headless

    def run(self) -> None:
        if self._headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption("Chess + Shogi AI")

        flags = pygame.RESIZABLE
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), flags)
        clock = pygame.time.Clock()

        menu = MenuScene(screen)
        # Seed menu with CLI-provided defaults so `--game shogi` opens on shogi.
        menu.config = self._initial_config

        scene: str = "menu"
        game_scene: GameScene | None = None
        running = True
        start_time = time.monotonic()

        while running:
            dt_ms = clock.tick(FPS)

            # Detect window resize between frames
            w, h = screen.get_size()
            # Dispatch input + update for the active scene
            if scene == "menu":
                for event in pygame.event.get():
                    if event.type == pygame.VIDEORESIZE:
                        menu.on_resize((w, h))
                        continue
                    action = menu.handle_event(event)
                    if action == MenuAction.QUIT:
                        running = False
                    elif action == MenuAction.START:
                        game_scene = GameScene(screen, menu.config)
                        scene = "game"
                        break
                if not running:
                    break
                menu.render(dt_ms)

            else:
                assert game_scene is not None
                for event in pygame.event.get():
                    if event.type == pygame.VIDEORESIZE:
                        game_scene.on_resize((w, h))
                        continue
                    action = game_scene.handle_event(event)
                    if action == GameSceneAction.QUIT:
                        running = False
                    elif action == GameSceneAction.BACK_TO_MENU:
                        game_scene.cleanup()
                        game_scene = None
                        scene = "menu"
                        break
                if not running or scene == "menu":
                    pygame.display.flip()
                    continue
                game_scene.process_engine_events()
                game_scene.render()

            pygame.display.flip()

            # Headless: quit after 2 seconds
            if self._headless and (time.monotonic() - start_time) >= 2.0:
                running = False

        if game_scene is not None:
            game_scene.cleanup()
        pygame.quit()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option("--game", default="chess", type=click.Choice(["chess", "shogi"]), show_default=True)
@click.option(
    "--engine", "engine_kind",
    default="rule", type=click.Choice(["rule", "model"]), show_default=True,
)
@click.option("--model-path", default="", help="Path to .safetensors weights (when --engine=model)")
@click.option(
    "--mode", "play_mode",
    default="human_ai",
    type=click.Choice(["ai_ai", "human_ai", "human_human"]),
    show_default=True,
)
@click.option(
    "--side", "human_side",
    default="white", type=click.Choice(["white", "black"]), show_default=True,
    help="Which side the human plays in human_ai mode",
)
@click.option("--headless", is_flag=True, default=False, help="Run without display (CI mode, exits after 2s)")
def main(
    game: str,
    engine_kind: str,
    model_path: str,
    play_mode: str,
    human_side: str,
    headless: bool,
) -> None:
    """Chess + Shogi AI GUI — powered by chess-mlx-gui."""
    config = GameConfig(
        game=game,  # type: ignore[arg-type]
        engine_kind=engine_kind,  # type: ignore[arg-type]
        model_path=model_path,
        play_mode=play_mode,  # type: ignore[arg-type]
        human_side=human_side,  # type: ignore[arg-type]
    )
    ChessApp(initial_config=config, headless=headless).run()


if __name__ == "__main__":
    main()
