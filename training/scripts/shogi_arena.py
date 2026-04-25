"""
Shogi arena harness for evaluating a trained ChessShogiTransformer checkpoint.

Plays matches between two players, reporting W/L/D from player-A's POV.
Supported player types:
    nn:CKPT_PATH        — NN greedy (pick highest-prob legal move)
    nn-sample:CKPT_PATH — NN sampling (proportional to softmax probabilities)
    random              — uniform random legal move

Examples::

    uv run python training/scripts/shogi_arena.py \\
        --a nn:checkpoints/shogi_40m/final.safetensors \\
        --b random \\
        --games 50 --max-plies 256

    uv run python training/scripts/shogi_arena.py \\
        --a nn:checkpoints/shogi_40m/final.safetensors \\
        --b nn:checkpoints/shogi_11m/final.safetensors \\
        --games 30
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


class RandomPlayer:
    """Uniform-random legal-move player."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select_move(self, board) -> int:
        moves = list(board.legal_moves)
        if not moves:
            return -1
        return moves[int(self.rng.integers(0, len(moves)))]


class NNPlayer:
    """NN-policy greedy or sampled player.

    Args:
        ckpt_path:   Path to a ``.safetensors`` checkpoint.
        n_layers:    Must match training config.
        d_model:     Must match training config.
        n_heads:     Must match training config.
        ffn_dim:     Must match training config.
        sample:      If True, sample from policy softmax over legal moves.
                     If False, take argmax (greedy).
        temperature: Softmax temperature for sampling (ignored when greedy).
        seed:        RNG seed for sampling.
    """

    def __init__(
        self,
        ckpt_path: str,
        n_layers: int = 12,
        d_model: int = 512,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        sample: bool = False,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        import mlx.core as mx  # noqa: PLC0415
        from training.models.transformer import ChessShogiTransformer  # noqa: PLC0415
        from training.data.dlshogi_loader import (  # noqa: PLC0415
            _encode_cshogi_position,
            _shogi_move_to_dlshogi_idx,
        )

        self._mx = mx
        self._encode = _encode_cshogi_position
        self._move_to_idx = _shogi_move_to_dlshogi_idx
        self.sample = sample
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = f"nn{'-sample' if sample else ''}:{Path(ckpt_path).name}"

        self.model = ChessShogiTransformer(
            game="shogi",
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
        )
        self.model.load_weights(ckpt_path)
        self.model.eval()

    def select_move(self, board) -> int:
        import cshogi  # noqa: PLC0415

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return -1
        if len(legal_moves) == 1:
            return legal_moves[0]

        mirror = (board.turn == cshogi.WHITE)
        x = self._encode(board, mirror)  # [81, 90]
        x_mx = self._mx.array(x[None, ...], dtype=self._mx.bfloat16)
        out = self.model(x_mx, game="shogi")
        logits = np.asarray(out["policy"].astype(self._mx.float32))[0]  # [2187]

        # Score each legal move by its policy logit
        legal_idxs: list[int] = []
        legal_logits: list[float] = []
        for mv in legal_moves:
            idx = self._move_to_idx(board, mv, mirror, cshogi)
            if 0 <= idx < logits.shape[0]:
                legal_idxs.append(idx)
                legal_logits.append(float(logits[idx]))
            else:
                legal_idxs.append(-1)
                legal_logits.append(-1e30)

        legal_logits_arr = np.asarray(legal_logits)

        if self.sample:
            T = max(1e-3, self.temperature)
            probs = np.exp((legal_logits_arr - legal_logits_arr.max()) / T)
            probs = probs / probs.sum()
            choice = int(self.rng.choice(len(legal_moves), p=probs))
        else:
            choice = int(np.argmax(legal_logits_arr))
        return legal_moves[choice]


def _make_player(spec: str, seed: int, **model_kw) -> object:
    spec = spec.strip()
    if spec == "random":
        return RandomPlayer(seed=seed)
    if spec.startswith("nn:") or spec.startswith("nn-sample:"):
        sample = spec.startswith("nn-sample:")
        path = spec.split(":", 1)[1]
        return NNPlayer(ckpt_path=path, sample=sample, seed=seed, **model_kw)
    raise ValueError(f"Unknown player spec: {spec!r}")


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def play_game(player_black, player_white, max_plies: int) -> str:
    """Play a single game; return 'B', 'W', or 'D'."""
    import cshogi  # noqa: PLC0415

    board = cshogi.Board()
    plies = 0

    while plies < max_plies:
        if board.is_game_over():
            break
        if board.is_draw():
            return "D"

        to_move = player_black if board.turn == cshogi.BLACK else player_white
        mv = to_move.select_move(board)
        if mv == -1 or not board.is_legal(mv):
            # Illegal or no-moves → side-to-move loses
            return "W" if board.turn == cshogi.BLACK else "B"
        board.push(mv)
        plies += 1

    if board.is_game_over():
        # Side-to-move was just mated by the previous move
        return "W" if board.turn == cshogi.BLACK else "B"
    return "D"  # max plies reached


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--a", "player_a_spec", required=True, help="Player A spec.")
@click.option("--b", "player_b_spec", required=True, help="Player B spec.")
@click.option("--games", type=int, default=20, show_default=True)
@click.option("--max-plies", type=int, default=256, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--swap/--no-swap", default=True, show_default=True,
              help="Swap colours every game so each player plays half black, half white.")
@click.option("--n-layers", type=int, default=12, show_default=True)
@click.option("--d-model", type=int, default=512, show_default=True)
@click.option("--n-heads", type=int, default=8, show_default=True)
@click.option("--ffn-dim", type=int, default=2048, show_default=True)
def main(
    player_a_spec: str,
    player_b_spec: str,
    games: int,
    max_plies: int,
    seed: int,
    swap: bool,
    n_layers: int,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
) -> None:
    model_kw = dict(
        n_layers=n_layers, d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim
    )
    a = _make_player(player_a_spec, seed=seed, **model_kw)
    b = _make_player(player_b_spec, seed=seed + 1, **model_kw)

    a_wins = 0
    b_wins = 0
    draws = 0
    plies_per_game: list[int] = []

    t0 = time.perf_counter()
    for g in range(games):
        if swap and g % 2 == 1:
            black_player, white_player = b, a
            label_for_black = "B"
        else:
            black_player, white_player = a, b
            label_for_black = "A"

        result = play_game(black_player, white_player, max_plies)

        if result == "D":
            draws += 1
            label = "D"
        elif (result == "B" and label_for_black == "A") or (result == "W" and label_for_black == "B"):
            a_wins += 1
            label = "A"
        else:
            b_wins += 1
            label = "B"

        elapsed = time.perf_counter() - t0
        click.echo(
            f"  game {g+1:3d}/{games} "
            f"({'A=B,B=W' if label_for_black=='A' else 'B=B,A=W'}): "
            f"winner={label}  cumul A:{a_wins} B:{b_wins} D:{draws}  "
            f"({elapsed:.1f}s)"
        )

    click.echo("=" * 60)
    click.echo(f"A: {a.name}")
    click.echo(f"B: {b.name}")
    click.echo(f"  Games: {games}  A wins: {a_wins}  B wins: {b_wins}  Draws: {draws}")
    p_a = (a_wins + 0.5 * draws) / games
    click.echo(f"  Score(A): {p_a:.3f}")
    if 0 < p_a < 1:
        import math  # noqa: PLC0415
        elo = -400 * math.log10(1.0 / p_a - 1.0)
        click.echo(f"  Elo diff (A-B, no error bars): {elo:+.0f}")


if __name__ == "__main__":
    main()
