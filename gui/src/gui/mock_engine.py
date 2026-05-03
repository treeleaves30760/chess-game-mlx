"""Mock engine that speaks UCI + JSON-RPC for GUI development and testing.

Run standalone:
    python -m gui.mock_engine
    python -m gui.mock_engine --game shogi

The mock engine:
- Responds to all standard UCI commands
- Generates random legal moves via python-chess
- Emits fake info lines every ~500 ms with increasing depth and random score
- Answers the get_eval_bar / get_top_moves JSON-RPC queries with random data
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from typing import Any

import chess
import chess.pgn

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _write(line: str) -> None:
    """Write a line to stdout and flush immediately."""
    print(line, flush=True)


def _write_json(obj: dict[str, Any]) -> None:
    _write(json.dumps(obj))


def _fake_pv(board: chess.Board, depth: int) -> list[str]:
    """Generate a random walk PV of up to *depth* moves."""
    pv: list[str] = []
    b = board.copy()
    for _ in range(min(depth, 8)):
        moves = list(b.legal_moves)
        if not moves:
            break
        move = random.choice(moves)
        pv.append(move.uci())
        b.push(move)
    return pv


def _fake_top_moves(board: chess.Board, k: int = 5) -> list[dict[str, Any]]:
    """Return k random legal moves with fake probabilities."""
    moves = list(board.legal_moves)
    random.shuffle(moves)
    chosen = moves[:k]
    raw_probs = [random.random() for _ in chosen]
    total = sum(raw_probs) or 1.0
    result = []
    for move, prob in zip(chosen, raw_probs, strict=True):
        result.append({"uci": move.uci(), "prob": round(prob / total, 3)})
    return sorted(result, key=lambda x: -x["prob"])


# ---------------------------------------------------------------------------
# Mock engine state
# ---------------------------------------------------------------------------


class MockEngine:
    """Self-contained UCI-speaking mock engine."""

    VERSION = "1.0"
    NAME = "MockEngine (chess-mlx-gui)"
    AUTHOR = "chess-mlx-gui test harness"

    def __init__(self, game: str = "chess") -> None:
        self._game = game
        self._board = chess.Board()
        self._searching = threading.Event()
        self._stop_search = threading.Event()
        self._search_thread: threading.Thread | None = None
        self._options: dict[str, str] = {
            "MultiPV": "1",
            "Hash": "512",
            "Threads": "4",
            "NN_Weights": "",
        }
        self._rpc_id_counter: int = 0

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Read stdin forever and dispatch commands."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                self._handle_jsonrpc(line)
            else:
                self._handle_uci(line)

    # -----------------------------------------------------------------------
    # UCI command handling
    # -----------------------------------------------------------------------

    def _handle_uci(self, line: str) -> None:
        tokens = line.split()
        if not tokens:
            return

        cmd = tokens[0]

        if cmd in ("uci", "usi"):
            _write(f"id name {self.NAME}")
            _write(f"id author {self.AUTHOR}")
            _write("option name MultiPV type spin default 1 min 1 max 10")
            _write("option name Hash type spin default 512 min 16 max 16384")
            _write("option name Threads type spin default 4 min 1 max 16")
            _write("option name NN_Weights type string default ")
            _write("uciok" if cmd == "uci" else "usiok")

        elif cmd == "isready":
            _write("readyok")

        elif cmd in ("ucinewgame", "usinewgame"):
            self._board = chess.Board()

        elif cmd == "position":
            self._parse_position(tokens[1:])

        elif cmd == "go":
            self._parse_go(tokens[1:])

        elif cmd == "stop":
            self._stop_search.set()

        elif cmd == "setoption":
            self._parse_setoption(tokens[1:])

        elif cmd == "quit":
            self._stop_search.set()
            sys.exit(0)

    def _parse_position(self, tokens: list[str]) -> None:
        """Handle 'position startpos [moves ...]' or 'position fen <FEN> [moves ...]'."""
        if not tokens:
            return
        self._board = chess.Board()

        if tokens[0] == "startpos":
            rest = tokens[1:]
        elif tokens[0] == "fen" and len(tokens) > 1:
            fen_parts = []
            i = 1
            while i < len(tokens) and tokens[i] != "moves":
                fen_parts.append(tokens[i])
                i += 1
            fen = " ".join(fen_parts)
            try:
                self._board = chess.Board(fen)
            except ValueError:
                self._board = chess.Board()
            rest = tokens[i:]
        else:
            rest = tokens

        if rest and rest[0] == "moves":
            for uci_move in rest[1:]:
                try:
                    move = chess.Move.from_uci(uci_move)
                    if move in self._board.legal_moves:
                        self._board.push(move)
                except ValueError:
                    pass

    def _parse_go(self, tokens: list[str]) -> None:
        """Handle 'go' command — start background search thread."""
        # Stop any existing search
        if self._search_thread and self._search_thread.is_alive():
            self._stop_search.set()
            self._search_thread.join(timeout=1.0)

        self._stop_search.clear()
        infinite = "infinite" in tokens

        # Parse movetime if present (in milliseconds)
        movetime_ms: int | None = None
        if "movetime" in tokens:
            try:
                mt_idx = tokens.index("movetime")
                movetime_ms = int(tokens[mt_idx + 1])
            except (ValueError, IndexError):
                pass

        board_snapshot = self._board.copy()
        self._search_thread = threading.Thread(
            target=self._search_loop,
            args=(board_snapshot, infinite, movetime_ms),
            daemon=True,
        )
        self._search_thread.start()

    def _parse_setoption(self, tokens: list[str]) -> None:
        """Handle 'setoption name X value Y'."""
        # tokens: ['name', 'MultiPV', 'value', '3']
        try:
            name_idx = tokens.index("name")
            if "value" in tokens:
                val_idx = tokens.index("value")
                name = " ".join(tokens[name_idx + 1: val_idx])
                value = " ".join(tokens[val_idx + 1:])
            else:
                name = " ".join(tokens[name_idx + 1:])
                value = ""
            self._options[name] = value
        except (ValueError, IndexError):
            pass

    # -----------------------------------------------------------------------
    # Search simulation
    # -----------------------------------------------------------------------

    def _search_loop(
        self,
        board: chess.Board,
        infinite: bool,
        movetime_ms: int | None = None,
    ) -> None:
        """Emit fake info lines, then bestmove."""
        multipv = int(self._options.get("MultiPV", "1"))
        score_base = random.randint(-60, 60)
        delay_s = 0.4  # emit info every 400ms

        start = time.monotonic()
        depth = 1

        while True:
            # Check for stop signal (non-blocking)
            stopped = self._stop_search.wait(timeout=delay_s)
            if stopped:
                break

            # Check movetime budget
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if movetime_ms is not None and elapsed_ms >= movetime_ms:
                break

            # Emit one info line per multipv
            nps = random.randint(40_000, 120_000)
            nodes = depth * nps // 2
            moves = list(board.legal_moves)
            random.shuffle(moves)

            for pv_idx in range(min(multipv, len(moves))):
                pv_move = moves[pv_idx]
                pv = [pv_move.uci(), *_fake_pv(board, depth - 1)]
                score_cp = score_base + random.randint(-20, 20) - pv_idx * 10
                _write(
                    f"info depth {depth} seldepth {depth + 2} multipv {pv_idx + 1} "
                    f"score cp {score_cp} nodes {nodes} nps {nps} "
                    f"time {int(elapsed_ms)} "
                    f"pv {' '.join(pv)}"
                )

            depth += 1
            if not infinite and movetime_ms is None and depth > 20:
                break

        # Emit bestmove
        legal = list(board.legal_moves)
        if legal:
            best = random.choice(legal)
            _write(f"bestmove {best.uci()}")
        else:
            _write("bestmove 0000")

    # -----------------------------------------------------------------------
    # JSON-RPC handling
    # -----------------------------------------------------------------------

    def _handle_jsonrpc(self, line: str) -> None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return

        method = obj.get("method", "")
        params = obj.get("params", {})
        req_id = obj.get("id")

        if method == "get_eval_bar":
            score_cp = random.randint(-80, 80)
            win_prob = 0.5 + score_cp / 1000.0
            win_prob = max(0.05, min(0.95, win_prob))
            if req_id is not None:
                _write_json({
                    "jsonrpc": "2.0",
                    "result": {
                        "score_cp": score_cp,
                        "win_prob": round(win_prob, 3),
                        "depth": random.randint(15, 25),
                    },
                    "id": req_id,
                })

        elif method == "get_top_moves":
            k = params.get("k", 3)
            top = _fake_top_moves(self._board, k)
            moves_info = []
            score_base = random.randint(-50, 50)
            for i, m in enumerate(top):
                b2 = self._board.copy()
                try:
                    move = chess.Move.from_uci(m["uci"])
                    if move in b2.legal_moves:
                        b2.push(move)
                except ValueError:
                    pass
                pv = _fake_pv(b2, 5)
                moves_info.append({
                    "uci": m["uci"],
                    "cp": score_base - i * 8 + random.randint(-5, 5),
                    "pv": [m["uci"], *pv],
                })
            if req_id is not None:
                _write_json({
                    "jsonrpc": "2.0",
                    "result": {"moves": moves_info},
                    "id": req_id,
                })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mock chess/shogi UCI engine")
    parser.add_argument("--game", default="chess", choices=["chess", "shogi"])
    args = parser.parse_args()

    engine = MockEngine(game=args.game)
    engine.run()


if __name__ == "__main__":
    main()
