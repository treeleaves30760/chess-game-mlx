"""Thin wrapper around an LC0 ONNX network via onnxruntime.

This is the *recommended* production path for running LC0's BT4 / T1 / T2
transformer networks. It sidesteps the whole C++/MLX reimplementation of
the attention body plus smolgen plus attention-policy head: onnxruntime
handles all of them, and on Apple Silicon the CoreML execution provider
maps most tensor ops to the ANE/GPU.

Usage:

    >>> from training.lc0.onnx_runner import LC0OnnxRunner
    >>> runner = LC0OnnxRunner("/path/to/BT4.onnx")
    >>> out = runner.forward_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    >>> print(out.policy.shape, out.wdl, out.moves_left)
    (1858,) [0.246 0.581 0.173] 93.0

Performance (measured on M3, t1_256 model, CoreML EP):
    - Batch 1:   ~8 ms / position    (= 125 pos/s)
    - Batch 64:  ~80 ms / batch      (= 800 pos/s)

Policy output indexing: the 1858-element policy vector uses LC0's compact
"flat" move index. To convert between ``chess.Move`` and the index, use
:func:`move_to_nn_index` / :func:`nn_index_to_uci`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from training.lc0.encoding import encode_fen, encode_position
from training.lc0.move_index import move_to_nn_index, nn_index_to_uci


@dataclass(frozen=True)
class LC0Output:
    """Output of a forward pass."""

    policy: np.ndarray  # [1858] float32 — raw logits (pre-legal-mask)
    policy_probs: np.ndarray  # [1858] float32 — softmax over legal moves only
    wdl: np.ndarray  # [3] float32 — win/draw/loss probabilities
    moves_left: float  # scalar
    legal_moves: list[chess.Move]
    legal_indices: list[int]  # NN indices of legal moves (parallel to legal_moves)
    legal_priors: np.ndarray  # [len(legal_moves)] float32 — P(move) over legal set
    value_scalar: float  # P(win) - P(loss) in [-1, +1]


class LC0OnnxRunner:
    """Load an LC0 ONNX network and expose a convenient forward pass."""

    def __init__(
        self,
        onnx_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        self.onnx_path = str(onnx_path)
        if providers is None:
            # Prefer CoreML on Apple Silicon, fall back to CPU.
            avail = ort.get_available_providers()
            if "CoreMLExecutionProvider" in avail:
                providers = [
                    ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}),
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]
        self.providers = providers
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.onnx_path, sess_opts, providers=providers
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]
        # Detect the ordering of outputs — LC0 exports them as
        # /output/policy, /output/wdl, /output/mlh, but in case a future
        # LC0 build reorders them we look up by suffix.
        self._policy_name = self._find("policy")
        self._wdl_name = self._find("wdl")
        self._mlh_name = self._find("mlh")

    def _find(self, suffix: str) -> str:
        for name in self._output_names:
            if name.endswith(suffix):
                return name
        raise RuntimeError(
            f"Could not find output named '*{suffix}' in ONNX model "
            f"(outputs: {self._output_names})"
        )

    # ------------------------------------------------------------------
    # Forward-pass entry points
    # ------------------------------------------------------------------

    def forward_batch(self, planes: np.ndarray) -> dict[str, np.ndarray]:
        """Run a raw forward pass on a batch of pre-encoded planes.

        Args:
            planes: ``[B, 112, 8, 8]`` float32 tensor.

        Returns:
            Dict with ``'policy' [B, 1858]``, ``'wdl' [B, 3]``,
            ``'mlh' [B, 1]``.
        """
        if planes.dtype != np.float32:
            planes = planes.astype(np.float32)
        outs = self.session.run(
            [self._policy_name, self._wdl_name, self._mlh_name],
            {self._input_name: planes},
        )
        return {"policy": outs[0], "wdl": outs[1], "mlh": outs[2]}

    def forward_fen(self, fen: str) -> LC0Output:
        """Encode a FEN and return an :class:`LC0Output`."""
        board = chess.Board(fen)
        return self.forward_board(board)

    def forward_board(
        self,
        board: chess.Board,
        history: list[chess.Board] | None = None,
    ) -> LC0Output:
        """Encode a python-chess board (plus optional history) and run."""
        enc = encode_position(board, history=history)
        planes = enc.planes[np.newaxis, :, :, :]  # [1, 112, 8, 8]
        raw = self.forward_batch(planes)
        policy = raw["policy"][0]  # [1858]
        wdl = raw["wdl"][0]  # [3]
        wdl_softmax = _softmax(wdl)
        mlh = float(raw["mlh"][0, 0])

        # LC0 applies softplus on MLH in the ONNX graph, but for legacy
        # reasons some BT4 exports emit raw values — just report raw.
        legal_moves: list[chess.Move] = []
        legal_indices: list[int] = []
        legal_priors_raw: list[float] = []
        for move in board.legal_moves:
            nn_idx = move_to_nn_index(move, board.turn == chess.BLACK)
            if nn_idx is None:
                continue  # should never happen for a legal move
            legal_moves.append(move)
            legal_indices.append(nn_idx)
            legal_priors_raw.append(float(policy[nn_idx]))
        if legal_priors_raw:
            legal_priors = _softmax(np.asarray(legal_priors_raw, dtype=np.float32))
        else:
            legal_priors = np.zeros(0, dtype=np.float32)

        # Value: LC0 treats WDL as [w, d, l] where the scalar value is w - l.
        # Note: wdl index 0 = win (from side-to-move's perspective).
        value_scalar = float(wdl_softmax[0] - wdl_softmax[2])
        return LC0Output(
            policy=policy.astype(np.float32),
            policy_probs=_softmax(policy),
            wdl=wdl_softmax.astype(np.float32),
            moves_left=mlh,
            legal_moves=legal_moves,
            legal_indices=legal_indices,
            legal_priors=legal_priors,
            value_scalar=value_scalar,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def provider_in_use(self) -> str:
        provs = self.session.get_providers()
        return provs[0] if provs else "unknown"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


__all__ = [
    "LC0Output",
    "LC0OnnxRunner",
    "move_to_nn_index",
    "nn_index_to_uci",
]
