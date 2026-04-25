"""
Export trained MLX weights to a flat safetensors file.

The exported bundle contains:
    1. All model weight tensors (float32 for C++ compatibility).
    2. A JSON sidecar ``<stem>.json`` with architecture config so C++ can
       validate the checkpoint before loading.

Usage::

    from training.export import export_model
    export_model(model, "checkpoints/run1_final.safetensors", game="chess")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from training.models.transformer import (
    CHESS_FEAT_DIM,
    CHESS_NUM_MOVES,
    CHESS_SEQ_LEN,
    DEFAULT_D_MODEL,
    DEFAULT_FFN_DIM,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    SHOGI_FEAT_DIM,
    SHOGI_NUM_MOVES,
    SHOGI_SEQ_LEN,
    ChessShogiTransformer,
)


def _mlx_to_numpy(arr: mx.array) -> np.ndarray:
    """Convert MLX array to float32 numpy array (materialises if needed)."""
    mx.eval(arr)
    return np.array(arr, dtype=np.float32)


def export_model(
    model: ChessShogiTransformer,
    output_path: str | Path,
    game: Literal["chess", "shogi"] | None = None,
    include_optimiser: bool = False,
) -> Path:
    """Export model weights to safetensors + JSON sidecar.

    Args:
        model:             Trained ``ChessShogiTransformer`` instance.
        output_path:       Target ``.safetensors`` file path.
        game:              Which game the model was trained for.  Falls back to
                           ``model.game`` if not supplied.
        include_optimiser: Include optimiser state (for resuming; default False).

    Returns:
        Path to the written ``.safetensors`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if game is None:
        g = model.game
        if g == "both":
            raise ValueError("Must specify game= when model.game == 'both'")
        game = g  # type: ignore[assignment]

    # ---------------------------------------------------------------
    # Force all pending evaluations before saving
    # ---------------------------------------------------------------
    from mlx.utils import tree_flatten  # noqa: PLC0415

    leaves = tree_flatten(model.parameters())
    mx.eval(*[arr for _, arr in leaves])

    # Save using MLX's built-in safetensors writer
    model.save_weights(str(output_path))

    # ---------------------------------------------------------------
    # Write JSON sidecar with architecture metadata
    # ---------------------------------------------------------------
    if game == "chess":
        seq_len = CHESS_SEQ_LEN
        feat_dim = CHESS_FEAT_DIM
        num_moves = CHESS_NUM_MOVES
    else:
        seq_len = SHOGI_SEQ_LEN
        feat_dim = SHOGI_FEAT_DIM
        num_moves = SHOGI_NUM_MOVES

    # Infer ffn_dim from the first layer's fc1 weight shape (output dim)
    try:
        first_layer = (model.chess_layers if game == "chess" else model.shogi_layers)[0]
        ffn_dim_actual = first_layer.ffn.fc1.weight.shape[0]
    except Exception:
        ffn_dim_actual = DEFAULT_FFN_DIM

    arch_config = {
        "version": "1.0",
        "game": game,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "ffn_dim": ffn_dim_actual,
        "seq_len": seq_len,
        "feat_dim": feat_dim,
        "num_moves": num_moves,
        "encoding_spec_version": "v1.0",
        "param_count": sum(v.size for _, v in leaves),
    }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(arch_config, f, indent=2)

    print(f"Exported weights → {output_path}")
    print(f"Exported config  → {json_path}")
    return output_path


def load_for_inference(
    weights_path: str | Path,
    game: Literal["chess", "shogi"] | None = None,
) -> ChessShogiTransformer:
    """Load a safetensors checkpoint for inference.

    Reads the JSON sidecar to reconstruct the architecture.

    Args:
        weights_path: Path to ``.safetensors`` file.
        game:         Override game (reads from sidecar by default).

    Returns:
        Loaded ``ChessShogiTransformer`` in eval mode.
    """
    weights_path = Path(weights_path)
    json_path = weights_path.with_suffix(".json")

    config: dict = {}
    if json_path.exists():
        with open(json_path) as f:
            config = json.load(f)

    g = game or config.get("game", "chess")
    n_layers = config.get("n_layers", DEFAULT_N_LAYERS)
    d_model = config.get("d_model", DEFAULT_D_MODEL)
    n_heads = config.get("n_heads", DEFAULT_N_HEADS)
    ffn_dim = config.get("ffn_dim", DEFAULT_FFN_DIM)

    model = ChessShogiTransformer(
        game=g,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
    )
    model.load_weights(str(weights_path))
    model.eval()
    return model
