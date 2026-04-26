"""
Shogi arena harness for evaluating a trained ChessShogiTransformer checkpoint.

Plays matches between two players, reporting W/L/D from player-A's POV.
Supported player types:
    nn:CKPT_PATH        — NN greedy (pick highest-prob legal move)
    nn-sample:CKPT_PATH — NN sampling (proportional to softmax probabilities)
    random              — uniform random legal move
    usi:BINARY_PATH     — any USI-protocol engine subprocess (dlshogi,
                          YaneuraOu, etc.); pass setoption values via
                          ``--a-setoption NAME=VALUE`` (repeatable)

Examples::

    uv run python training/scripts/shogi_arena.py \\
        --a nn:checkpoints/shogi_40m/final.safetensors \\
        --b random \\
        --games 50 --max-plies 256

    # NN vs NN sanity check
    uv run python training/scripts/shogi_arena.py \\
        --a nn:checkpoints/shogi_40m/final.safetensors \\
        --b nn:checkpoints/shogi_11m/final.safetensors \\
        --games 30

    # Ground-truth Elo: our XGD model vs dlshogi (license: personal/research only)
    uv run python training/scripts/shogi_arena.py \\
        --a nn:checkpoints/shogi_xgd/final.safetensors \\
        --b usi:/path/to/dlshogi \\
        --b-setoption DNN_Model=data/dlshogi_nets/model-dr2_exhi.onnx \\
        --b-setoption Threads=1 \\
        --b-movetime 1000 \\
        --games 20 --max-plies 256
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


class UsiPlayer:
    """USI-protocol subprocess player (dlshogi, YaneuraOu, etc.).

    Holds a persistent subprocess across moves so the engine's network /
    transposition table / search state is preserved per game. ``movetime_ms``
    bounds each move; alternatively ``nodes`` caps node count.

    Reset between games is via ``usinewgame``.
    """

    def __init__(
        self,
        engine_path: str,
        setoptions: list[tuple[str, str]] | None = None,
        movetime_ms: int = 1000,
        nodes: int | None = None,
        warmup_timeout: float = 90.0,
        per_move_timeout: float = 30.0,
        name: str | None = None,
    ) -> None:
        import os  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        if not os.access(engine_path, os.X_OK):
            raise FileNotFoundError(f"USI engine not executable: {engine_path}")

        self._engine_path = engine_path
        self._movetime_ms = movetime_ms
        self._nodes = nodes
        self._per_move_timeout = per_move_timeout
        self.name = name or os.path.basename(engine_path)

        self._proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("usi")
        self._read_until("usiok", timeout_seconds=warmup_timeout)
        for opt_name, opt_value in setoptions or []:
            self._send(f"setoption name {opt_name} value {opt_value}")
        self._send("isready")
        self._read_until("readyok", timeout_seconds=warmup_timeout)
        self._send("usinewgame")
        # Warmup eval — first ONNX/Network init can take 30+ s.
        self._send("position startpos")
        self._send(f"go {self._go_clause()}")
        self._read_until("bestmove", timeout_seconds=warmup_timeout)

    def _send(self, cmd: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read_until(self, needle: str, timeout_seconds: float = 30.0) -> list[str]:
        import time  # noqa: PLC0415
        assert self._proc.stdout is not None
        out: list[str] = []
        t0 = time.time()
        while time.time() - t0 < timeout_seconds:
            line = self._proc.stdout.readline()
            if not line:
                break
            out.append(line.rstrip())
            if line.startswith(needle):
                return out
        raise RuntimeError(
            f"{self.name} didn't emit '{needle}' in {timeout_seconds}s; "
            f"last: {out[-3:] if out else '<none>'}"
        )

    def _go_clause(self) -> str:
        if self._nodes is not None:
            return f"nodes {self._nodes}"
        return f"movetime {self._movetime_ms}"

    def reset_for_new_game(self) -> None:
        self._send("usinewgame")
        self._send("isready")
        self._read_until("readyok", timeout_seconds=10.0)

    def select_move(self, board) -> int:
        self._send(f"position sfen {board.sfen()}")
        self._send(f"go {self._go_clause()}")
        try:
            lines = self._read_until("bestmove", timeout_seconds=self._per_move_timeout)
        except RuntimeError:
            return -1
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] not in ("resign", "win", "(none)"):
                    try:
                        # Board.move_from_usi is an instance method; we parse
                        # against the *current* board state to validate legality.
                        mv = board.move_from_usi(parts[1])
                        return int(mv) if mv else -1
                    except Exception:
                        return -1
                return -1
        return -1

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


def _make_player(
    spec: str,
    seed: int,
    setoptions: list[tuple[str, str]] | None = None,
    movetime_ms: int = 1000,
    usi_nodes: int | None = None,
    **model_kw,
) -> object:
    spec = spec.strip()
    if spec == "random":
        return RandomPlayer(seed=seed)
    if spec.startswith("nn:") or spec.startswith("nn-sample:"):
        sample = spec.startswith("nn-sample:")
        path = spec.split(":", 1)[1]
        return NNPlayer(ckpt_path=path, sample=sample, seed=seed, **model_kw)
    if spec.startswith("usi:"):
        path = spec.split(":", 1)[1]
        return UsiPlayer(
            engine_path=path,
            setoptions=setoptions,
            movetime_ms=movetime_ms,
            nodes=usi_nodes,
            name=Path(path).name,
        )
    raise ValueError(f"Unknown player spec: {spec!r}")


def _parse_setoption_pairs(raw: tuple[str, ...]) -> list[tuple[str, str]]:
    """Parse repeated --x-setoption NAME=VALUE flags into [(name, value), ...]."""
    out: list[tuple[str, str]] = []
    for item in raw:
        if "=" not in item:
            raise click.BadParameter(
                f"--*-setoption expects 'NAME=VALUE', got: {item!r}"
            )
        name, value = item.split("=", 1)
        out.append((name.strip(), value.strip()))
    return out


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
@click.option(
    "--a-setoption", "a_setoption", multiple=True,
    help="Repeatable. USI setoption for player A in 'NAME=VALUE' form (only for usi: spec).",
)
@click.option(
    "--b-setoption", "b_setoption", multiple=True,
    help="Repeatable. USI setoption for player B in 'NAME=VALUE' form (only for usi: spec).",
)
@click.option("--a-movetime", type=int, default=1000, show_default=True,
              help="Per-move time budget (ms) for usi: spec player A.")
@click.option("--b-movetime", type=int, default=1000, show_default=True,
              help="Per-move time budget (ms) for usi: spec player B.")
@click.option("--a-nodes", type=int, default=None,
              help="Per-move node budget for usi: spec player A (overrides --a-movetime).")
@click.option("--b-nodes", type=int, default=None,
              help="Per-move node budget for usi: spec player B (overrides --b-movetime).")
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
    a_setoption: tuple[str, ...],
    b_setoption: tuple[str, ...],
    a_movetime: int,
    b_movetime: int,
    a_nodes: int | None,
    b_nodes: int | None,
) -> None:
    model_kw = dict(
        n_layers=n_layers, d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim
    )
    a = _make_player(
        player_a_spec, seed=seed,
        setoptions=_parse_setoption_pairs(a_setoption),
        movetime_ms=a_movetime, usi_nodes=a_nodes,
        **model_kw,
    )
    b = _make_player(
        player_b_spec, seed=seed + 1,
        setoptions=_parse_setoption_pairs(b_setoption),
        movetime_ms=b_movetime, usi_nodes=b_nodes,
        **model_kw,
    )

    a_wins = 0
    b_wins = 0
    draws = 0
    plies_per_game: list[int] = []

    def _reset(p: object) -> None:
        if hasattr(p, "reset_for_new_game"):
            p.reset_for_new_game()  # type: ignore[attr-defined]

    def _close(p: object) -> None:
        if hasattr(p, "close"):
            try:
                p.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    t0 = time.perf_counter()
    try:
        for g in range(games):
            if swap and g % 2 == 1:
                black_player, white_player = b, a
                label_for_black = "B"
            else:
                black_player, white_player = a, b
                label_for_black = "A"

            _reset(black_player)
            _reset(white_player)
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
    finally:
        _close(a)
        _close(b)

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
