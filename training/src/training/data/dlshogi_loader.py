"""
dlshogi-compatible data loader for shogi positions.

Currently provides a **synthetic** loader that generates random shogi positions
using python-shogi and assigns random action-values.  The real dlshogi teacher
data loader can be slotted in later with the same batch interface.

Batch format (same convention as chessbench_loader):
    positions:      [B, 81, 90]  bf16
    policy_target:  [B, 2187]    float32 soft labels / one-hot
    value_target:   [B, 1]       float32 in [-1, +1]
    moves_left:     [B, 1]       float32 ≥ 0
"""

from __future__ import annotations

import math
import random
from collections.abc import Generator, Iterator
from typing import NamedTuple

import mlx.core as mx
import numpy as np

SHOGI_NUM_MOVES = 2187
SHOGI_SEQ_LEN = 81
SHOGI_FEAT_DIM = 90

# ---------------------------------------------------------------------------
# Fast cshogi → 90-plane encoder (no python-shogi round-trip)
# ---------------------------------------------------------------------------
# cshogi piece values:
#   1..14   = black side: PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD, KING,
#             PROM_PAWN, PROM_LANCE, PROM_KNIGHT, PROM_SILVER, PROM_BISHOP, PROM_ROOK
#   17..30  = white side: same piece-type ordering, offset by 16
# Encoding spec plane order (own side, planes 0..13):
#   歩 香 桂 銀 角 飛 金 玉 と 成香 成桂 成銀 馬 龍
# Mapping cshogi piece-type (1..14) → plane (0..13):
_PT_TO_PLANE = np.zeros(15, dtype=np.int32)
# cshogi piece-type ↔ encoding-spec plane
_PT_TO_PLANE[1] = 0   # PAWN  → 歩
_PT_TO_PLANE[2] = 1   # LANCE → 香
_PT_TO_PLANE[3] = 2   # KNIGHT→ 桂
_PT_TO_PLANE[4] = 3   # SILVER→ 銀
_PT_TO_PLANE[5] = 4   # BISHOP→ 角
_PT_TO_PLANE[6] = 5   # ROOK  → 飛
_PT_TO_PLANE[7] = 6   # GOLD  → 金
_PT_TO_PLANE[8] = 7   # KING  → 玉
_PT_TO_PLANE[9] = 8   # PROM_PAWN  → と
_PT_TO_PLANE[10] = 9  # PROM_LANCE → 成香
_PT_TO_PLANE[11] = 10 # PROM_KNIGHT→ 成桂
_PT_TO_PLANE[12] = 11 # PROM_SILVER→ 成銀
_PT_TO_PLANE[13] = 12 # PROM_BISHOP→ 馬
_PT_TO_PLANE[14] = 13 # PROM_ROOK  → 龍

# Hand-piece order in cshogi.Board.pieces_in_hand: (歩 香 桂 銀 金 角 飛)
_HAND_MAX = np.array([18, 4, 4, 4, 4, 2, 2], dtype=np.float32)


def _encode_cshogi_position(cb, mirror: bool) -> np.ndarray:
    """Encode a ``cshogi.Board`` into a [81, 90] float32 tensor.

    Bit-exact with :func:`training.data.encoding.encode_shogi_position` when the
    same SFEN is re-parsed via python-shogi.  ``mirror`` should be set to
    ``cb.turn == cshogi.WHITE``: when 後手 is to move, the board is rotated 180°
    and colours are swapped so the mover is always at the bottom.
    """
    out = np.zeros((SHOGI_SEQ_LEN, SHOGI_FEAT_DIM), dtype=np.float32)

    # --- Piece planes 0..27 (own 0-13, opponent 14-27) ---
    pieces = np.asarray(cb.pieces, dtype=np.int16)  # [81], 0 = empty
    nz = pieces != 0
    sqs = np.nonzero(nz)[0]
    if sqs.size > 0:
        pvs = pieces[sqs]
        is_white = pvs >= 17
        pts = pvs - np.where(is_white, 16, 0)  # 1..14

        if mirror:
            enc_sqs = 80 - sqs
            is_own = is_white  # own side is WHITE when mirroring
        else:
            enc_sqs = sqs
            is_own = ~is_white

        plane_offset = np.where(is_own, 0, 14)
        planes = plane_offset + _PT_TO_PLANE[pts]
        out[enc_sqs, planes] = 1.0

    # --- Hand planes 56..69 ---
    own_hand = cb.pieces_in_hand[1] if mirror else cb.pieces_in_hand[0]
    opp_hand = cb.pieces_in_hand[0] if mirror else cb.pieces_in_hand[1]
    own_arr = np.asarray(own_hand, dtype=np.float32) / _HAND_MAX
    opp_arr = np.asarray(opp_hand, dtype=np.float32) / _HAND_MAX
    out[:, 56:63] = own_arr  # broadcast over 81 squares
    out[:, 63:70] = opp_arr

    # --- Plane 89: side-to-move (1 if 先手) ---
    if not mirror:  # equiv. to cb.turn == BLACK
        out[:, 89] = 1.0

    return out


# ---------------------------------------------------------------------------
# Fast cshogi-move → dlshogi 2187-slot index
# ---------------------------------------------------------------------------
# 81 squares × 27 move types = 2187
# Move types 0..9 : 10 directions (no promote)
# Move types 10..19: 10 directions with promote
# Move types 20..26: drops (歩 香 桂 銀 金 角 飛)
#
# Direction map matches encoding.py shogi_move_to_idx:
#   (df, dr) for unit step in board (file_delta, rank_delta from black's view)
#   0:( 0,-1) Forward   1:( 1,-1) FR    2:(-1,-1) FL
#   3:( 0, 1) Backward  4:( 1, 1) BR    5:(-1, 1) BL
#   6:( 1, 0) Right     7:(-1, 0) Left
#   8:( 1,-2) KnightFR  9:(-1,-2) KnightFL


def _shogi_move_to_dlshogi_idx(cb, cmove: int, mirror: bool, cshogi_mod) -> int:
    """Map a cshogi move int to a 2187-slot dlshogi policy index.

    ``mirror`` should be ``cb.turn == cshogi.WHITE`` (i.e. the same flag used
    when encoding the position).  Returns ``-1`` if the move falls outside the
    schema — caller should skip such records.
    """
    is_drop = cshogi_mod.move_is_drop(cmove)
    promo = cshogi_mod.move_is_promotion(cmove)
    to_sq = cshogi_mod.move_to(cmove)
    enc_to = (80 - to_sq) if mirror else to_sq

    if is_drop:
        # cshogi drop_hand_piece is 0..6 in the same order as pieces_in_hand
        # (歩=0, 香=1, 桂=2, 銀=3, 金=4, 角=5, 飛=6)
        hand_idx = cshogi_mod.move_drop_hand_piece(cmove)
        if not (0 <= hand_idx < 7):
            return -1
        move_type = 20 + hand_idx
        return enc_to * 27 + move_type

    from_sq = cshogi_mod.move_from(cmove)
    enc_from = (80 - from_sq) if mirror else from_sq

    # Compute (df, dr) on the encoded board
    # Square index in encoding spec: sq = (9-file)*9 + (rank-1)
    # → file = 9 - sq // 9; rank = sq % 9 + 1
    from_file = 9 - enc_from // 9
    from_rank = enc_from % 9 + 1
    to_file = 9 - enc_to // 9
    to_rank = enc_to % 9 + 1
    df = to_file - from_file
    dr = to_rank - from_rank

    # Knight jump check first (only legal shogi knight delta from black's POV)
    if abs(df) == 1 and dr == -2:
        dir_idx = 8 if df == 1 else 9
    else:
        # Normalise to unit direction
        if df == 0:
            unit_df = 0
        else:
            unit_df = 1 if df > 0 else -1
        if dr == 0:
            unit_dr = 0
        else:
            unit_dr = 1 if dr > 0 else -1
        _DIR_MAP = {
            (0, -1): 0, (1, -1): 1, (-1, -1): 2,
            (0, 1): 3, (1, 1): 4, (-1, 1): 5,
            (1, 0): 6, (-1, 0): 7,
        }
        dir_idx = _DIR_MAP.get((unit_df, unit_dr), -1)
        if dir_idx < 0:
            return -1

    move_type = dir_idx + (10 if promo else 0)
    return enc_from * 27 + move_type


class ShogiBatch(NamedTuple):
    """A single mini-batch of shogi training data as MLX arrays."""

    positions: mx.array      # [B, 81, 90]  bf16
    policy_target: mx.array  # [B, 2187]    float32
    value_target: mx.array   # [B, 1]       float32
    moves_left: mx.array     # [B, 1]       float32


def _try_encode_shogi():  # type: ignore[return]
    """Import shogi encoder; fall back to None if python-shogi unavailable."""
    try:
        from training.data.encoding import encode_shogi_position  # noqa: PLC0415
        return encode_shogi_position
    except ImportError:
        return None


def synthetic_shogi_batch_generator(
    batch_size: int = 32,
    seed: int | None = None,
    max_batches: int | None = None,
) -> Generator[ShogiBatch, None, None]:
    """Infinite (or bounded) generator of synthetic shogi training batches.

    If python-shogi is not installed, generates random tensors with correct
    shapes so the training pipeline can still be tested end-to-end.

    Args:
        batch_size:  Positions per batch.
        seed:        Optional RNG seed.
        max_batches: Stop after this many batches (``None`` = infinite).

    Yields:
        ``ShogiBatch`` namedtuples.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    encode_fn = _try_encode_shogi()

    batches_produced = 0

    while max_batches is None or batches_produced < max_batches:
        if encode_fn is not None:
            try:
                import shogi  # type: ignore[import]  # noqa: PLC0415

                positions_list: list[np.ndarray] = []
                policy_indices: list[int] = []
                values: list[float] = []
                moves_left_vals: list[float] = []

                while len(positions_list) < batch_size:
                    board = shogi.Board()
                    for _ in range(rng.randint(0, 60)):
                        moves = list(board.legal_moves)
                        if not moves or board.is_game_over():
                            break
                        mv = rng.choice(moves)
                        board.push(mv)

                    positions_list.append(encode_fn(board))
                    policy_indices.append(rng.randint(0, SHOGI_NUM_MOVES - 1))
                    values.append(float(np_rng.uniform(-1.0, 1.0)))
                    moves_left_vals.append(float(np_rng.uniform(1, 200)))

                positions_np = np.stack(positions_list[:batch_size])
                policy_np = np.zeros((batch_size, SHOGI_NUM_MOVES), dtype=np.float32)
                for i, idx in enumerate(policy_indices[:batch_size]):
                    policy_np[i, idx] = 1.0
                value_np = np.array([[v] for v in values[:batch_size]], dtype=np.float32)
                ml_np = np.array([[m] for m in moves_left_vals[:batch_size]], dtype=np.float32)

            except (ImportError, Exception):
                # Fallback: pure random tensors
                positions_np = np_rng.random((batch_size, 81, 90)).astype(np.float32)
                policy_np = np.zeros((batch_size, SHOGI_NUM_MOVES), dtype=np.float32)
                rand_idx = np_rng.integers(0, SHOGI_NUM_MOVES, size=batch_size)
                for i, idx in enumerate(rand_idx):
                    policy_np[i, idx] = 1.0
                value_np = np_rng.uniform(-1.0, 1.0, (batch_size, 1)).astype(np.float32)
                ml_np = np_rng.uniform(1, 200, (batch_size, 1)).astype(np.float32)
        else:
            # python-shogi not installed — random tensors
            positions_np = np_rng.random((batch_size, 81, 90)).astype(np.float32)
            policy_np = np.zeros((batch_size, SHOGI_NUM_MOVES), dtype=np.float32)
            rand_idx = np_rng.integers(0, SHOGI_NUM_MOVES, size=batch_size)
            for i, idx in enumerate(rand_idx):
                policy_np[i, idx] = 1.0
            value_np = np_rng.uniform(-1.0, 1.0, (batch_size, 1)).astype(np.float32)
            ml_np = np_rng.uniform(1, 200, (batch_size, 1)).astype(np.float32)

        yield ShogiBatch(
            positions=mx.array(positions_np, dtype=mx.bfloat16),
            policy_target=mx.array(policy_np),
            value_target=mx.array(value_np),
            moves_left=mx.array(ml_np),
        )
        batches_produced += 1


class DlshogiLoader:
    """Real loader for dlshogi-format ``.hcpe`` files.

    Each record in an ``.hcpe`` file is a ``cshogi.HuffmanCodedPosAndEval``:
        hcp:        32-byte huffman-encoded position
        eval:       int16 centipawn from the *black* (先手) point of view
        bestMove16: int16 PSV-format best move
        gameResult: int8  cshogi BLACK_WIN / WHITE_WIN / DRAW

    The loader maps each record to a :class:`ShogiBatch` element using the
    encoding spec already in use by the rest of the project
    (90-plane positions, 2187-slot policy, ``[-1, +1]`` value).

    Conversions:
        positions      = encode_shogi_position(python-shogi Board re-parsed
                         from cshogi SFEN).  This automatically applies the
                         dlshogi convention of mirroring the board so the
                         side-to-move is always at the bottom.
        policy_target  = one-hot at shogi_move_to_idx(best_move) using the
                         (post-mirror) python-shogi Board.
        value_target   = ``value_lambda * tanh(eval/600) + (1-value_lambda) * outcome``
                         in [-1, +1], **from the side-to-move perspective**
                         (sign-flipped when 後手 is to move).
        moves_left     = 0 (hcpe has no per-position remaining-ply info; we
                         keep the moves-left head loss-weight low so this
                         is harmless).

    Args:
        path:           Path to a single ``.hcpe`` file *or* a directory
                        containing one or more ``.hcpe`` files (concatenated).
        batch_size:     Positions per batch.
        seed:           RNG seed for shuffling.
        value_lambda:   Mix weight: 1.0 = pure eval-based value, 0.0 = pure
                        outcome-based value.  Default 0.5 is the dlshogi
                        recommendation.
        max_batches:    Stop after this many batches (``None`` = unlimited).
        skip_invalid:   If True, silently skip records where the best move
                        falls outside the 2187-slot schema (rare).  If False
                        such records yield an all-zero policy target.
    """

    def __init__(
        self,
        path: str,
        batch_size: int = 256,
        seed: int = 42,
        value_lambda: float = 0.5,
        max_batches: int | None = None,
        skip_invalid: bool = True,
    ) -> None:
        import os  # noqa: PLC0415

        self.batch_size = batch_size
        self.seed = seed
        self.value_lambda = value_lambda
        self.max_batches = max_batches
        self.skip_invalid = skip_invalid

        # Lazy import cshogi to avoid hard dep at module load
        import cshogi  # noqa: PLC0415

        self._cshogi = cshogi
        self._dtype = cshogi.HuffmanCodedPosAndEval

        # Resolve path → list of files
        if os.path.isdir(path):
            files = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".hcpe")
            )
            if not files:
                raise FileNotFoundError(f"No .hcpe files found in directory: {path}")
        else:
            files = [path]

        # mmap-load each file as a structured numpy array, then concat views
        # (np.memmap concat would copy; we instead materialise the index
        # space and lookup per-batch — keeps RAM low while shuffling fast).
        self._mmaps: list[np.memmap] = []
        offsets: list[int] = [0]
        for fp in files:
            mm = np.memmap(fp, dtype=self._dtype, mode="r")
            self._mmaps.append(mm)
            offsets.append(offsets[-1] + len(mm))
        self._cum_offsets = np.array(offsets, dtype=np.int64)
        self._total = int(offsets[-1])

        if self._total == 0:
            raise ValueError(f"hcpe file(s) appear empty: {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def _record_at(self, idx: int) -> np.ndarray:
        """Return the structured record at the global index ``idx``."""
        # Find which file
        file_idx = int(np.searchsorted(self._cum_offsets, idx, side="right")) - 1
        local_idx = idx - int(self._cum_offsets[file_idx])
        return self._mmaps[file_idx][local_idx]

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[ShogiBatch]:
        cshogi = self._cshogi
        rng = np.random.default_rng(self.seed)
        batches_yielded = 0

        positions_buf = np.zeros((self.batch_size, SHOGI_SEQ_LEN, SHOGI_FEAT_DIM), dtype=np.float32)
        policy_buf = np.zeros((self.batch_size, SHOGI_NUM_MOVES), dtype=np.float32)
        value_buf = np.zeros((self.batch_size, 1), dtype=np.float32)
        ml_buf = np.zeros((self.batch_size, 1), dtype=np.float32)

        # Reuse a single Board to avoid Python-side allocation per record
        cb = cshogi.Board()

        while True:
            if self.max_batches is not None and batches_yielded >= self.max_batches:
                return

            policy_buf.fill(0.0)

            # Oversample indices to cover skips
            idxs = rng.integers(0, self._total, size=self.batch_size * 2)

            filled = 0
            i = 0
            while filled < self.batch_size and i < len(idxs):
                rec = self._record_at(int(idxs[i]))
                i += 1

                cb.set_hcp(rec["hcp"])
                mirror = (cb.turn == cshogi.WHITE)

                # Best-move idx — compute first so we can skip invalid records
                # without paying the position-encoding cost.
                m16 = int(rec["bestMove16"])
                if m16 == 0:
                    if self.skip_invalid:
                        continue
                    pol_idx = -1
                else:
                    cmove = cb.move_from_psv(m16)
                    pol_idx = _shogi_move_to_dlshogi_idx(cb, cmove, mirror, cshogi)
                    if pol_idx < 0 and self.skip_invalid:
                        continue

                positions_buf[filled] = _encode_cshogi_position(cb, mirror)

                if 0 <= pol_idx < SHOGI_NUM_MOVES:
                    policy_buf[filled, pol_idx] = 1.0

                # Value combines eval (cp from black POV) + gameResult.
                eval_cp = float(rec["eval"])
                value_eval = math.tanh(eval_cp / 600.0)  # black POV
                gr = int(rec["gameResult"])
                if gr == cshogi.BLACK_WIN:
                    value_outcome = 1.0
                elif gr == cshogi.WHITE_WIN:
                    value_outcome = -1.0
                else:
                    value_outcome = 0.0

                if mirror:  # convert to side-to-move POV
                    value_eval = -value_eval
                    value_outcome = -value_outcome

                value_buf[filled, 0] = (
                    self.value_lambda * value_eval
                    + (1.0 - self.value_lambda) * value_outcome
                )
                ml_buf[filled, 0] = 0.0

                filled += 1

            if filled == 0:
                # Pathological — every sample invalid. Yield zeros to keep
                # downstream shapes stable; trainer will record a short loss.
                positions_buf.fill(0.0)
                value_buf.fill(0.0)
            elif filled < self.batch_size:
                for j in range(filled, self.batch_size):
                    positions_buf[j] = positions_buf[0]
                    policy_buf[j] = policy_buf[0]
                    value_buf[j] = value_buf[0]
                    ml_buf[j] = ml_buf[0]

            yield ShogiBatch(
                positions=mx.array(positions_buf, dtype=mx.bfloat16),
                policy_target=mx.array(policy_buf),
                value_target=mx.array(value_buf),
                moves_left=mx.array(ml_buf),
            )
            batches_yielded += 1


def make_shogi_loader(
    synthetic: bool = True,
    data_dir: str | None = None,
    batch_size: int = 256,
    seed: int = 42,
    max_batches: int | None = None,
    value_lambda: float = 0.5,
) -> Generator[ShogiBatch, None, None] | Iterator[ShogiBatch]:
    """Create a shogi data loader.

    Args:
        synthetic:    If True, returns the random synthetic generator
                      (no real data needed).  Otherwise the real
                      :class:`DlshogiLoader` is used.
        data_dir:     Path to ``.hcpe`` file or directory of ``.hcpe`` files.
        batch_size:   Positions per batch.
        seed:         RNG seed.
        max_batches:  Stop after this many batches (``None`` = unlimited).
        value_lambda: Eval-vs-outcome mix when ``synthetic=False``
                      (dlshogi recommends 0.5).

    Returns:
        An iterable of :class:`ShogiBatch`.
    """
    if synthetic:
        return synthetic_shogi_batch_generator(
            batch_size=batch_size, seed=seed, max_batches=max_batches
        )
    if data_dir is None:
        raise ValueError("data_dir must be set when synthetic=False")
    return DlshogiLoader(
        path=data_dir,
        batch_size=batch_size,
        seed=seed,
        value_lambda=value_lambda,
        max_batches=max_batches,
    )
