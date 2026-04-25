"""LC0 112-plane input encoding (``INPUT_CLASSICAL_112_PLANE``).

This is a faithful port of ``src/neural/encoder.cc`` in the LC0 source tree.
The output tensor has shape ``[112, 8, 8]`` (float32, NCHW), which matches the
layout expected by LC0's ONNX networks (input node ``/input/planes``).

Plane layout (``kPlanesPerBoard`` = 13, ``kMoveHistory`` = 8):

    0..5     our {pawns, knights, bishops, rooks, queens, kings}
    6..11    their {pawns, knights, bishops, rooks, queens, kings}
    12       repetitions (0/1 flag, all-ones if GetRepetitions >= 1)
    ...
    91..103  history ply 7 (board state 7 plies ago, if available)
    104      queenside-castling (us) — all-ones plane if true
    105      kingside-castling (us) — all-ones
    106      queenside-castling (them) — all-ones
    107      kingside-castling (them) — all-ones
    108      black-to-move flag — all-ones if side-to-move is black
    109      Rule50 counter (half-moves since last capture/pawn push), raw value
    110      all zeros (legacy move-count plane, zero for non-armageddon formats)
    111      all ones (edge-finding helper plane)

Bit-to-square mapping (matches LC0's ``BitBoard``): bit ``i`` corresponds to
``row = i // 8`` (rank, 0 = rank 1), ``col = i % 8`` (file, 0 = file a).

When side-to-move is black, LC0 calls ``board.Mirror()`` (see
``src/chess/board.cc:1029``) so the "ours"/"theirs" semantics are always from
the perspective of the side to move, and ranks are flipped (rank 8 ↔ rank 1).
This module reproduces that behavior.

The ``HistoryFill`` setting defaults to ``FEN_ONLY`` in LC0's UCI — for FEN
positions without move history, LC0 fills all 8 history plies with the
current position (unless the current position is the standard startpos, in
which case empty history planes are left at zero). We match that default.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

# -----------------------------------------------------------------------------
# Constants (match ``src/neural/encoder.h``)
# -----------------------------------------------------------------------------
K_PLANES_PER_BOARD = 13
K_MOVE_HISTORY = 8
K_AUX_PLANE_BASE = K_PLANES_PER_BOARD * K_MOVE_HISTORY  # 104
K_INPUT_PLANES = K_AUX_PLANE_BASE + 8  # 112

# Piece-type → index inside the 6-tuple
_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


@dataclass(frozen=True)
class EncodedPosition:
    """Result of encoding: input tensor plus the transform applied."""

    planes: np.ndarray  # shape [112, 8, 8], dtype float32
    transform: int = 0  # always 0 for INPUT_CLASSICAL_112_PLANE


def _bitboard_to_plane(bb: int) -> np.ndarray:
    """Convert a 64-bit bitboard to an 8x8 float32 plane.

    LC0 indexes bits as ``a1=0, b1=1, ..., h8=63``, so bit ``i`` maps to
    square ``chess.SQUARES[i]``. When reshaped to ``[8, 8]`` the first axis
    is rank (0 = rank 1) and the second axis is file (0 = file a).
    """
    plane = np.zeros(64, dtype=np.float32)
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        plane[idx] = 1.0
        bb ^= lsb
    return plane.reshape(8, 8)


def _all_ones_plane() -> np.ndarray:
    return np.ones((8, 8), dtype=np.float32)


def _scalar_plane(value: float) -> np.ndarray:
    return np.full((8, 8), value, dtype=np.float32)


def _mirror_board(board: chess.Board) -> chess.Board:
    """Flip the board vertically (rank 1↔rank 8) AND swap piece colors.

    Equivalent to LC0's ``ChessBoard::Mirror()``. python-chess's
    :meth:`chess.Board.mirror` does exactly this.
    """
    return board.mirror()


def _encode_one_board(
    board: chess.Board,
    *,
    repetitions: int,
    result: np.ndarray,
    base: int,
    en_passant_undo_idx: int | None = None,
) -> None:
    """Fill 13 planes at ``result[base:base+13]`` for a single (possibly
    mirrored) board snapshot.

    Args:
        board:       A python-chess board representing the position from the
                     perspective of the side to move (i.e. already mirrored
                     if black was to move originally).
        repetitions: Number of prior occurrences of this position.
        result:      Output tensor ``[112, 8, 8]``.
        base:        Starting plane index (``i * 13``).
        en_passant_undo_idx: If not None, apply LC0's "undo pawn push" trick
            using this as the stored en-passant bit index on LC0's internal
            board (values 0..7 mean "our" pawn was pushed from rank 2→4,
            56..63 mean "their" pawn was pushed from rank 7→5). Used for
            synthetic history plies when we know the previous move was a
            double pawn push.
    """
    # "Our" pieces = pieces of color chess.WHITE on the (possibly mirrored)
    # board. Because we always mirror when the original side to move was
    # black, "WHITE" here always means "side to move".
    our_color = chess.WHITE
    their_color = chess.BLACK
    our_masks = {
        pt: int(board.pieces_mask(pt, our_color)) for pt in _PIECE_ORDER
    }
    their_masks = {
        pt: int(board.pieces_mask(pt, their_color)) for pt in _PIECE_ORDER
    }

    if en_passant_undo_idx is not None:
        idx = en_passant_undo_idx
        if idx < 8:
            # "Us" pawn was on rank 4 (bit idx+24) — move it back to rank 2
            # (bit idx+8). Operate bit-wise to sidestep Python's
            # arbitrary-precision int (LC0's C++ uses unsigned wraparound
            # to get the same result in one add).
            our_masks[chess.PAWN] |= 1 << (idx + 8)
            our_masks[chess.PAWN] &= ~(1 << (idx + 24))
        else:
            # "Their" pawn on rank 5 → rank 7.
            file = idx - 56
            their_masks[chess.PAWN] |= 1 << (file + 48)
            their_masks[chess.PAWN] &= ~(1 << (file + 32))

    for piece_idx, piece_type in enumerate(_PIECE_ORDER):
        result[base + piece_idx] = _bitboard_to_plane(our_masks[piece_type])
        result[base + 6 + piece_idx] = _bitboard_to_plane(their_masks[piece_type])
    if repetitions >= 1:
        result[base + 12] = _all_ones_plane()


def _castling_rights_from_mirrored(board: chess.Board) -> tuple[bool, bool, bool, bool]:
    """Return ``(we_000, we_00, they_000, they_00)`` for a board whose "WHITE"
    side is the side to move (i.e. already mirrored)."""
    # On the mirrored board, "us" is WHITE.
    cast = board.castling_rights
    we_00 = bool(cast & chess.BB_H1)  # kingside rook on h1
    we_000 = bool(cast & chess.BB_A1)
    they_00 = bool(cast & chess.BB_H8)
    they_000 = bool(cast & chess.BB_A8)
    return we_000, we_00, they_000, they_00


def _compute_ep_undo_idx(board: chess.Board) -> int | None:
    """Compute LC0's stored en-passant bit index for its internal (already
    mirrored if black-to-move) board representation.

    LC0's convention: the e.p. bit is always stored on the 8th rank of the
    *side that just made the pushing move*, i.e. on LC0's ``kRank8`` as seen
    from the opposite side's perspective. After mirroring (for black-to-move
    positions), the bit ends up on rank 8 (bits 56..63) iff the pusher was
    "them" (i.e. the current side-to-move captures with a normal pawn
    advance), and on rank 1 (bits 0..7) iff the pusher was "us" (in a
    canonical-format position where we_are_black is expressed differently,
    which never occurs for our INPUT_CLASSICAL_112_PLANE format).

    Consequence: for INPUT_CLASSICAL_112_PLANE inputs (which is what all
    BT/T transformer nets use), the undo always applies to "their" pawns.
    The idx is always in ``[56, 63]``.
    """
    if board.ep_square is None:
        return None
    file = chess.square_file(board.ep_square)
    return 56 + file


def encode_position(
    board: chess.Board,
    *,
    history: list[chess.Board] | None = None,
    fill_empty_history: str = "fen_only",
) -> EncodedPosition:
    """Encode a single chess position into 112 planes.

    Args:
        board: Current position. Any value of ``board.turn`` is supported.
        history: Optional list of prior boards (most recent last). If shorter
            than 7, earlier plies are either left zero or filled with the
            oldest-known board (``fill_empty_history`` determines which).
        fill_empty_history: One of ``"no"``, ``"fen_only"`` (LC0 default),
            ``"always"``. Matches LC0's ``HistoryFill`` option.

    Returns:
        :class:`EncodedPosition` with ``planes`` shape ``[112, 8, 8]``.
    """
    if fill_empty_history not in ("no", "fen_only", "always"):
        raise ValueError(f"fill_empty_history must be one of no/fen_only/always")
    history = list(history or [])
    # Build full position list, newest-last. LC0 iterates newest→oldest.
    full_history: list[chess.Board] = [*history, board]
    we_are_black = board.turn == chess.BLACK

    result = np.zeros((K_INPUT_PLANES, 8, 8), dtype=np.float32)

    # -------- Aux planes (indices 104..111) --------
    # The castling semantics: from "our" perspective (i.e. from side-to-move's
    # point of view). So look at the original (unmirrored) board's castling
    # rights for the side to move.
    # If we_are_black, "we" = black and "they" = white.
    if we_are_black:
        we_000 = board.has_queenside_castling_rights(chess.BLACK)
        we_00 = board.has_kingside_castling_rights(chess.BLACK)
        they_000 = board.has_queenside_castling_rights(chess.WHITE)
        they_00 = board.has_kingside_castling_rights(chess.WHITE)
    else:
        we_000 = board.has_queenside_castling_rights(chess.WHITE)
        we_00 = board.has_kingside_castling_rights(chess.WHITE)
        they_000 = board.has_queenside_castling_rights(chess.BLACK)
        they_00 = board.has_kingside_castling_rights(chess.BLACK)
    if we_000:
        result[K_AUX_PLANE_BASE + 0] = _all_ones_plane()
    if we_00:
        result[K_AUX_PLANE_BASE + 1] = _all_ones_plane()
    if they_000:
        result[K_AUX_PLANE_BASE + 2] = _all_ones_plane()
    if they_00:
        result[K_AUX_PLANE_BASE + 3] = _all_ones_plane()

    # Plane 108: black-to-move flag (only for non-canonical formats)
    if we_are_black:
        result[K_AUX_PLANE_BASE + 4] = _all_ones_plane()

    # Plane 109: raw rule50 counter
    result[K_AUX_PLANE_BASE + 5] = _scalar_plane(float(board.halfmove_clock))

    # Plane 110: legacy movecount plane (now always zero for
    # INPUT_CLASSICAL_112_PLANE — left as zeros)

    # Plane 111: all ones
    result[K_AUX_PLANE_BASE + 7] = _all_ones_plane()

    # -------- History planes (indices 0..103 in chunks of 13) --------
    # We go from newest (i=0 → planes 0..12) to oldest (i=7 → planes 91..103).
    # LC0 flips the board each time we step backwards by one ply (bool flip
    # flag). The "flip" here accounts for alternating side to move. We also
    # always present the board from the current side-to-move's perspective —
    # so we mirror once up front if we_are_black, and then flip alternates as
    # we walk backwards.
    n_history = len(full_history)
    flip = False
    ep_undo = _compute_ep_undo_idx(board)
    # LC0 stores the e.p. bit on the *current* board's pawn bitboard (as a
    # bit outside ``kPawnMask``). When iterating history backwards, the
    # undo fires whenever ``history_idx < 0`` AND the reconstructed board
    # still carries that bit. Since our python-chess history boards don't
    # preserve this non-standard storage, we check the current-board e.p.
    # up-front and apply the undo for every history_idx < 0 plane.
    for i in range(K_MOVE_HISTORY):
        # Which original board to use at history step i (0 = most recent)
        history_idx = n_history - 1 - i
        if history_idx < 0:
            # "Older than provided history"
            if fill_empty_history == "no":
                break
            if fill_empty_history == "fen_only":
                # If the oldest known board is the *standard* startpos, stop
                # (matches LC0's ``position.GetBoard() == kStartposBoard``
                # check). Any other FEN — including e.g. the position after
                # 1.e4 — is allowed to pad history with the oldest board.
                oldest = full_history[0]
                if oldest == chess.Board():
                    break
                # Otherwise, pad with the oldest known board.
                orig_board = full_history[0]
            else:  # "always"
                orig_board = full_history[0]
        else:
            orig_board = full_history[history_idx]

        # Apply the accumulated mirror state. This matches LC0's logic:
        # every time history_idx > 0, we toggle the `flip` flag AFTER
        # writing the planes for the current i. So for i=0 we write the
        # board unflipped relative to the current position's orientation;
        # since the "current position's orientation" in LC0 is such that
        # "our" pieces are the side-to-move's pieces, we must mirror the
        # original board iff we_are_black for i=0.
        snapshot = orig_board.copy(stack=False)
        should_mirror = we_are_black ^ flip
        if should_mirror:
            snapshot = _mirror_board(snapshot)

        # Repetitions: python-chess has board.is_repetition(). For simplicity
        # we default to 0 here (history not tracked). That's fine for
        # standalone FEN evaluation.
        repetitions = 0
        undo_idx = ep_undo if history_idx < 0 else None
        _encode_one_board(
            snapshot,
            repetitions=repetitions,
            result=result,
            base=i * K_PLANES_PER_BOARD,
            en_passant_undo_idx=undo_idx,
        )

        # Toggle flip for the next older ply
        if history_idx > 0:
            flip = not flip

    return EncodedPosition(planes=result, transform=0)


def encode_fen(
    fen: str,
    *,
    fill_empty_history: str = "fen_only",
) -> EncodedPosition:
    """Encode a FEN into 112 planes (no move history)."""
    board = chess.Board(fen)
    return encode_position(board, history=None, fill_empty_history=fill_empty_history)
