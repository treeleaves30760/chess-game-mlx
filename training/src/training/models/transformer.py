"""
ChessShogiTransformer — BT4-inspired encoder-only Post-LN transformer.

Target: ~40 M parameters on 24 GB M3 (trains comfortably at batch 128-256 bf16).

Architecture:
    - 12 transformer layers
    - d_model = 512, n_heads = 8, ffn_dim = 2048
    - Post-LayerNorm (DeepNet style)
    - Smolgen dynamic attention-bias module per layer
    - Trainable positional embeddings
    - Mish activation in FFN blocks
    - DeepNet initialisation (α, β scaling)
    - Three output heads: policy, value, moves-left

Input shapes (float32 or bfloat16):
    - Chess:  [B, 64, 19]
    - Shogi:  [B, 81, 90]

References:
    - LC0 BT4: https://lczero.org/blog/2024/02/transformer-progress/
    - DeepNet: https://arxiv.org/abs/2203.00555
"""

from __future__ import annotations

import math
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from training.models.heads import MovesLeftHead, PolicyHead, ValueHead
from training.models.smolgen import Smolgen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHESS_SEQ_LEN: int = 64
CHESS_FEAT_DIM: int = 19
CHESS_NUM_MOVES: int = 1858

SHOGI_SEQ_LEN: int = 81
SHOGI_FEAT_DIM: int = 90
SHOGI_NUM_MOVES: int = 2187

# Default model size — 40 M ± 3 M target
DEFAULT_N_LAYERS: int = 12
DEFAULT_D_MODEL: int = 512
DEFAULT_N_HEADS: int = 8
DEFAULT_FFN_DIM: int = 2048


# ---------------------------------------------------------------------------
# Mish activation
# ---------------------------------------------------------------------------

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.tanh(nn.softplus(x))


# ---------------------------------------------------------------------------
# Multi-head self-attention with Smolgen bias
# ---------------------------------------------------------------------------

class MultiHeadSelfAttentionSmolgen(nn.Module):
    """Post-LN multi-head self-attention augmented with Smolgen dynamic bias.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        seq_len: Sequence length (64 chess / 81 shogi).
        smolgen_inner: Smolgen inner (compression) size.
        smolgen_gen: Smolgen generation size.
        alpha: DeepNet α scaling applied to residual sub-layer output.
        dropout: Attention dropout probability (default 0 — disabled at test).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        smolgen_inner: int = 32,
        smolgen_gen: int = 256,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.alpha = alpha
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.smolgen = Smolgen(d_model, seq_len, n_heads, smolgen_inner, smolgen_gen)

        if dropout > 0.0:
            self.dropout: nn.Dropout | None = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, S, _ = x.shape

        # Compute dynamic bias before projection (uses pre-norm activations)
        smolgen_bias = self.smolgen(x)  # [B, n_heads, S, S]

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # [B, n_heads, S, S]
        attn_logits = (q @ k.transpose(0, 1, 3, 2)) / self.scale + smolgen_bias

        if mask is not None:
            attn_logits = attn_logits + mask

        attn_weights = mx.softmax(attn_logits, axis=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # [B, n_heads, S, head_dim] → [B, S, d_model]
        attn_out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Post-LN residual with DeepNet α scaling
        return self.norm(x + self.alpha * attn_out)


# ---------------------------------------------------------------------------
# FFN block with Mish
# ---------------------------------------------------------------------------

class FFNBlock(nn.Module):
    """Post-LN FFN block with Mish activation.

    Args:
        d_model: Hidden dimension.
        ffn_dim: Inner FFN dimension (typically 4× d_model).
        alpha: DeepNet α scaling on residual.
        dropout: Dropout after activation (default 0).
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.act = Mish()
        self.norm = nn.LayerNorm(d_model)
        self.dropout: nn.Dropout | None = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.act(self.fc1(x))
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(x + self.alpha * h)


# ---------------------------------------------------------------------------
# Single transformer layer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """One Post-LN transformer layer (attention + FFN)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        seq_len: int,
        smolgen_inner: int = 32,
        smolgen_gen: int = 256,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttentionSmolgen(
            d_model, n_heads, seq_len, smolgen_inner, smolgen_gen, alpha, dropout
        )
        self.ffn = FFNBlock(d_model, ffn_dim, alpha, dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.attn(x, mask)
        return self.ffn(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ChessShogiTransformer(nn.Module):
    """BT4-inspired encoder-only transformer for chess and shogi.

    Parameters are shared across the backbone; only the input projection and
    policy head are game-specific.  Value and moves-left heads are shared
    (the sign convention from encoding_spec handles side-to-move differences).

    Args:
        game:        Which game to instantiate heads for.  ``"both"`` creates
                     both policy heads (useful for inspection; ``forward()``
                     still requires a game argument).
        n_layers:    Number of transformer layers (default 12).
        d_model:     Hidden dimension (default 512).
        n_heads:     Attention heads (default 8).
        ffn_dim:     FFN inner dimension (default 2048).
        dropout:     Dropout probability (default 0.0).
        smolgen_inner: Smolgen compression size (default 32).
        smolgen_gen:   Smolgen generation size (default 256).
    """

    def __init__(
        self,
        game: Literal["chess", "shogi", "both"] = "chess",
        n_layers: int = DEFAULT_N_LAYERS,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        ffn_dim: int = DEFAULT_FFN_DIM,
        dropout: float = 0.0,
        smolgen_inner: int = 16,
        smolgen_gen: int = 16,
    ) -> None:
        super().__init__()

        self.game = game
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads

        # ---------------------------------------------------------------
        # DeepNet α, β initialisation
        # N = n_layers in the DeepNet paper; each layer has one attention
        # sub-layer and one FFN sub-layer → 2N sub-layers.
        # ---------------------------------------------------------------
        N = n_layers
        alpha = float((2 * N) ** 0.25)
        beta = float((8 * N) ** -0.25)

        # ---------------------------------------------------------------
        # Game-specific input projections
        # ---------------------------------------------------------------
        if game in ("chess", "both"):
            self.chess_input_proj = nn.Linear(CHESS_FEAT_DIM, d_model)
            self.chess_pos_embed = nn.Embedding(CHESS_SEQ_LEN, d_model)
        if game in ("shogi", "both"):
            self.shogi_input_proj = nn.Linear(SHOGI_FEAT_DIM, d_model)
            self.shogi_pos_embed = nn.Embedding(SHOGI_SEQ_LEN, d_model)

        # ---------------------------------------------------------------
        # Shared backbone — one list of layers; seq_len is a runtime value
        # (64 for chess, 81 for shogi).  Smolgen is seq_len-dependent, so
        # we create separate layer stacks per game when game=="both".
        # For single-game mode we create one stack.
        # ---------------------------------------------------------------
        if game in ("chess", "both"):
            self.chess_layers: list[TransformerLayer] = [
                TransformerLayer(
                    d_model, n_heads, ffn_dim, CHESS_SEQ_LEN,
                    smolgen_inner, smolgen_gen, alpha, dropout,
                )
                for _ in range(n_layers)
            ]
        if game in ("shogi", "both"):
            self.shogi_layers: list[TransformerLayer] = [
                TransformerLayer(
                    d_model, n_heads, ffn_dim, SHOGI_SEQ_LEN,
                    smolgen_inner, smolgen_gen, alpha, dropout,
                )
                for _ in range(n_layers)
            ]

        self.final_norm = nn.LayerNorm(d_model)

        # ---------------------------------------------------------------
        # Output heads
        # ---------------------------------------------------------------
        if game in ("chess", "both"):
            self.chess_policy_head = PolicyHead(d_model, CHESS_NUM_MOVES)
        if game in ("shogi", "both"):
            self.shogi_policy_head = PolicyHead(d_model, SHOGI_NUM_MOVES)

        # Value and moves-left heads are shared (game-agnostic)
        self.value_head = ValueHead(d_model)
        self.moves_left_head = MovesLeftHead(d_model)

        # Apply DeepNet β initialisation to all weight matrices
        self._apply_deepnet_init(beta)

    # -------------------------------------------------------------------
    # Initialisation helpers
    # -------------------------------------------------------------------

    def _apply_deepnet_init(self, beta: float) -> None:
        """Scale output projections of attention + FFN by β (DeepNet init)."""
        layers_to_scale: list[nn.Module] = []
        if self.game in ("chess", "both"):
            layers_to_scale.extend(self.chess_layers)
        if self.game in ("shogi", "both"):
            layers_to_scale.extend(self.shogi_layers)

        for layer in layers_to_scale:
            # Scale out_proj of attention
            w = layer.attn.out_proj.weight
            layer.attn.out_proj.weight = w * beta
            # Scale second FC of FFN
            w2 = layer.ffn.fc2.weight
            layer.ffn.fc2.weight = w2 * beta

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def __call__(
        self,
        x: mx.array,
        game: Literal["chess", "shogi"] | None = None,
    ) -> dict[str, mx.array]:
        """Forward pass.

        Args:
            x:    Input tensor.
                  Chess: ``[B, 64, 19]``, Shogi: ``[B, 81, 90]``.
            game: Which game.  If not supplied, falls back to ``self.game``
                  (only valid when ``self.game != "both"``).

        Returns:
            Dictionary with keys ``"policy"`` `[B, num_moves]`,
            ``"value"`` `[B, 1]`, ``"moves_left"`` `[B, 1]`.
        """
        if game is None:
            if self.game == "both":
                raise ValueError("Must pass game= when model was created with game='both'")
            game = self.game  # type: ignore[assignment]

        if game == "chess":
            B, S, _ = x.shape
            # Input projection + positional embedding
            h = self.chess_input_proj(x)  # [B, 64, d_model]
            positions = mx.arange(S)
            h = h + self.chess_pos_embed(positions)  # broadcast over B
            # Transformer layers
            for layer in self.chess_layers:
                h = layer(h)
            h = self.final_norm(h)
            policy = self.chess_policy_head(h)

        elif game == "shogi":
            B, S, _ = x.shape
            h = self.shogi_input_proj(x)  # [B, 81, d_model]
            positions = mx.arange(S)
            h = h + self.shogi_pos_embed(positions)
            for layer in self.shogi_layers:
                h = layer(h)
            h = self.final_norm(h)
            policy = self.shogi_policy_head(h)

        else:
            raise ValueError(f"Unknown game: {game!r}")

        value = self.value_head(h)
        moves_left = self.moves_left_head(h)

        return {"policy": policy, "value": value, "moves_left": moves_left}

    # -------------------------------------------------------------------
    # Parameter count utility
    # -------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable scalar parameters."""
        from mlx.utils import tree_flatten  # noqa: PLC0415

        leaves = tree_flatten(self.parameters())
        return sum(v.size for _, v in leaves)
