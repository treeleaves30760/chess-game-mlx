"""Engine client: communicates with a chess/shogi engine subprocess.

Supports:
- Standard UCI / USI text commands
- JSON-RPC extension notifications (lines starting with '{')
- Thread-safe event queue polled by the main Pygame loop each frame

Usage:
    client = EngineClient("mock")          # start built-in mock engine
    client = EngineClient("/path/to/eng")  # start real engine binary
    client.start()
    client.send_uci("uci")
    for event in client.poll_events():
        handle(event)
    client.stop()
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Default chess_engine binary location (relative to repo root).
# Resolved once at import time; the GUI runs from any cwd.
# --------------------------------------------------------------------------
def _default_chess_engine_path() -> str | None:
    """Return absolute path to engine/bin/chess_engine if it exists, else None."""
    here = Path(__file__).resolve()
    # gui/src/gui/engine_client.py → repo root is parents[3]
    for parent in here.parents:
        candidate = parent / "engine" / "bin" / "chess_engine"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


_DEFAULT_CHESS_ENGINE = _default_chess_engine_path()

# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UciOkEvent:
    """Engine replied uciok / usiok."""
    engine_name: str = ""
    engine_author: str = ""


@dataclass
class ReadyOkEvent:
    """Engine replied readyok."""
    pass


@dataclass
class InfoUpdate:
    """Parsed UCI info line."""

    depth: int = 0
    seldepth: int = 0
    multipv: int = 1
    score_cp: int = 0
    score_mate: int | None = None
    nodes: int = 0
    nps: int = 0
    time_ms: int = 0
    pv: list[str] = field(default_factory=list)
    hashfull: int = 0


@dataclass
class BestMoveEvent:
    """Engine emitted bestmove."""
    move: str = ""
    ponder: str | None = None


@dataclass
class EvalBarResult:
    """Response to get_eval_bar JSON-RPC call."""
    score_cp: int = 0
    win_prob: float = 0.5
    depth: int = 0
    request_id: int = 0


@dataclass
class TopMovesResult:
    """Response to get_top_moves JSON-RPC call."""
    moves: list[dict[str, Any]] = field(default_factory=list)
    request_id: int = 0


@dataclass
class EngineError:
    """Something went wrong."""
    message: str = ""


EngineEvent = (
    UciOkEvent
    | ReadyOkEvent
    | InfoUpdate
    | BestMoveEvent
    | EvalBarResult
    | TopMovesResult
    | EngineError
)


# ---------------------------------------------------------------------------
# UCI info line parser
# ---------------------------------------------------------------------------


def parse_info_line(line: str) -> InfoUpdate | None:  # noqa: PLR0912, PLR0915
    """Parse a UCI 'info ...' line into an InfoUpdate.

    Returns None if the line is not an info line.
    """
    tokens = line.split()
    if not tokens or tokens[0] != "info":
        return None

    update = InfoUpdate()
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "depth" and i + 1 < len(tokens):
            try:
                update.depth = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "seldepth" and i + 1 < len(tokens):
            try:
                update.seldepth = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "multipv" and i + 1 < len(tokens):
            try:
                update.multipv = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "score" and i + 1 < len(tokens):
            kind = tokens[i + 1]
            if kind == "cp" and i + 2 < len(tokens):
                try:
                    update.score_cp = int(tokens[i + 2])
                except ValueError:
                    pass
                i += 3
            elif kind == "mate" and i + 2 < len(tokens):
                try:
                    update.score_mate = int(tokens[i + 2])
                except ValueError:
                    pass
                i += 3
            else:
                i += 2
        elif tok == "nodes" and i + 1 < len(tokens):
            try:
                update.nodes = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "nps" and i + 1 < len(tokens):
            try:
                update.nps = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "time" and i + 1 < len(tokens):
            try:
                update.time_ms = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "hashfull" and i + 1 < len(tokens):
            try:
                update.hashfull = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "pv":
            update.pv = tokens[i + 1:]
            break
        else:
            i += 1

    return update


# ---------------------------------------------------------------------------
# Engine client
# ---------------------------------------------------------------------------


class EngineClient:
    """Manages a chess/shogi engine subprocess with UCI + JSON-RPC.

    Thread model:
    - Main thread:     calls send_uci / send_jsonrpc / poll_events
    - Reader thread:   reads stdout lines, parses, pushes to _event_queue
    """

    MOCK_SENTINEL = "mock"

    def __init__(self, path_or_mock: str = MOCK_SENTINEL, game: str = "chess") -> None:
        self._path = path_or_mock
        self._game = game
        self._process: subprocess.Popen[str] | None = None
        self._event_queue: queue.Queue[EngineEvent] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._rpc_id: int = 0
        self._stopped = threading.Event()
        self._uci_ok = threading.Event()
        self._engine_name = "unknown"
        self._pending_rpc: dict[int, str] = {}  # id -> method

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the engine subprocess and reader thread.

        The `path_or_mock` constructor argument is interpreted as follows:
          * ``"mock"`` → launch the built-in mock engine (uses python-chess)
          * path ending in ``.onnx`` → launch ``chess_engine --lc0-weights <path>``
            (LC0 BT4-style network via ONNX Runtime)
          * path ending in ``.safetensors`` → launch ``chess_engine --weights <path>``
            (our native ChessShogiTransformer)
          * any other path → treat as a UCI-compatible binary, launch with no args
        For the auto-detected weight cases, ``--threads 4`` is added by default.
        """
        if self._path == self.MOCK_SENTINEL:
            cmd = [sys.executable, "-m", "gui.mock_engine", "--game", self._game]
        else:
            path_lower = self._path.lower()
            if path_lower.endswith((".onnx", ".safetensors")):
                if _DEFAULT_CHESS_ENGINE is None:
                    raise RuntimeError(
                        "GUI received a weights file but couldn't locate "
                        "engine/bin/chess_engine. Build the C++ engine first: "
                        "`cmake --build build -j` from the repo root."
                    )
                flag = "--lc0-weights" if path_lower.endswith(".onnx") else "--weights"
                cmd = [_DEFAULT_CHESS_ENGINE, flag, self._path, "--threads", "4"]
            else:
                cmd = [self._path]

        self._stopped.clear()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="engine-reader"
        )
        self._reader_thread.start()

    def stop(self) -> None:
        """Send quit and shut down."""
        self._stopped.set()
        if self._process is not None:
            try:
                self.send_uci("quit")
                self._process.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def wait_for_uciok(self, timeout: float = 5.0) -> bool:
        """Block until uciok / usiok is received. Returns True on success."""
        return self._uci_ok.wait(timeout=timeout)

    # -----------------------------------------------------------------------
    # Sending
    # -----------------------------------------------------------------------

    def send_uci(self, line: str) -> None:
        """Send a raw UCI/USI line to the engine."""
        if self._process is None or self._process.stdin is None:
            return
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except OSError as e:
            # Engine subprocess is dead and we can't write to it any more.
            # Surface this so the GUI doesn't silently keep "trying" to drive
            # an engine that is gone.
            self._event_queue.put(
                EngineError(message=f"send to engine failed: {e}")
            )

    def send_jsonrpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        notification: bool = False,
    ) -> int:
        """Send a JSON-RPC request. Returns the request id (0 for notifications)."""
        if notification:
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
            self.send_uci(json.dumps(payload))
            return 0

        self._rpc_id += 1
        rid = self._rpc_id
        self._pending_rpc[rid] = method
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": rid,
        }
        self.send_uci(json.dumps(payload))
        return rid

    # -----------------------------------------------------------------------
    # Polling
    # -----------------------------------------------------------------------

    def poll_events(self) -> list[EngineEvent]:
        """Drain the event queue and return all pending events (non-blocking)."""
        events: list[EngineEvent] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    # -----------------------------------------------------------------------
    # Background reader
    # -----------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Background thread: read stdout lines and push parsed events."""
        assert self._process is not None
        assert self._process.stdout is not None

        for raw_line in self._process.stdout:
            if self._stopped.is_set():
                break
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            self._dispatch_line(line)

        # The for-loop ends when stdout closes — typically because the engine
        # process exited. If we didn't ask it to stop, surface an error so the
        # GUI can show "engine died" instead of silently sitting forever
        # waiting for a bestmove that will never arrive.
        if not self._stopped.is_set():
            rc = self._process.poll() if self._process else None
            self._event_queue.put(
                EngineError(message=f"engine process exited (rc={rc})")
            )

    def _dispatch_line(self, line: str) -> None:
        """Parse and enqueue an event for a single stdout line."""
        # JSON-RPC branch
        if line.startswith("{"):
            self._handle_json_line(line)
            return

        # Standard UCI / USI text
        tokens = line.split()
        if not tokens:
            return

        if tokens[0] in ("uciok", "usiok"):
            self._uci_ok.set()
            self._event_queue.put(
                UciOkEvent(engine_name=self._engine_name)
            )

        elif tokens[0] == "readyok":
            self._event_queue.put(ReadyOkEvent())

        elif tokens[0] == "id":
            if len(tokens) >= 3 and tokens[1] == "name":
                self._engine_name = " ".join(tokens[2:])

        elif tokens[0] == "info":
            update = parse_info_line(line)
            if update is not None:
                self._event_queue.put(update)

        elif tokens[0] == "bestmove":
            move = tokens[1] if len(tokens) > 1 else ""
            ponder = None
            if len(tokens) >= 4 and tokens[2] == "ponder":
                ponder = tokens[3]
            self._event_queue.put(BestMoveEvent(move=move, ponder=ponder))

    def _handle_json_line(self, line: str) -> None:
        """Parse a JSON-RPC line from the engine."""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            self._event_queue.put(EngineError(f"JSON decode error: {line[:80]}"))
            return

        # Notification (no 'id')
        if "id" not in obj:
            method = obj.get("method", "")
            params = obj.get("params", {})
            event = self._parse_notification(method, params)
            if event is not None:
                self._event_queue.put(event)
            return

        # Response to a request we sent
        rid = obj.get("id")
        method = self._pending_rpc.pop(rid, "unknown")
        if "error" in obj:
            self._event_queue.put(
                EngineError(f"RPC error ({method}): {obj['error']}")
            )
            return

        result = obj.get("result", {})
        if method == "get_eval_bar":
            self._event_queue.put(
                EvalBarResult(
                    score_cp=result.get("score_cp", 0),
                    win_prob=result.get("win_prob", 0.5),
                    depth=result.get("depth", 0),
                    request_id=rid,
                )
            )
        elif method == "get_top_moves":
            self._event_queue.put(
                TopMovesResult(
                    moves=result.get("moves", []),
                    request_id=rid,
                )
            )

    def _parse_notification(self, method: str, params: dict[str, Any]) -> EngineEvent | None:
        return None

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
