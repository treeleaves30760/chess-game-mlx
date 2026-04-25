"""
Position encoding — Python implementation matching encoding_spec.md v1.0.

Both chess and shogi encoders must be bit-exact with the C++ counterparts.
All operations are pure integer / boolean — no floating-point arithmetic —
so float32 outputs are guaranteed identical across platforms.

Chess encoding (§1):
    Output shape: [64, 19]  float32
    Square ordering: a1=0, b1=1, …, h1=7, a2=8, …, h8=63  (python-chess default)
    Feature planes 0-11: piece placement (white P/N/B/R/Q/K, black P/N/B/R/Q/K)
    Feature 12: side-to-move (broadcast)
    Features 13-16: castling rights (broadcast)
    Feature 17: en-passant target square
    Feature 18: half-move clock / 100.0 (broadcast)

Shogi encoding (§3):
    Output shape: [81, 90]  float32
    Square ordering: (9 - file) * 9 + (rank - 1), matching cshogi convention.
    Features follow dlshogi standard (planes 0-27 active in v1, rest zero).
    Side-to-move handling: board is mirrored (rotated 180° + colour-swapped)
    when 後手 (white) is to move, so "own side" is always at the bottom.
"""

from __future__ import annotations

import numpy as np
import chess

# ---------------------------------------------------------------------------
# Chess encoding
# ---------------------------------------------------------------------------

# python-chess piece type integers: PAWN=1 … KING=6
# Feature plane layout (same as encoding_spec Table 1)
_WHITE_PIECE_PLANES: dict[int, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
_BLACK_PIECE_PLANES: dict[int, int] = {
    chess.PAWN: 6,
    chess.KNIGHT: 7,
    chess.BISHOP: 8,
    chess.ROOK: 9,
    chess.QUEEN: 10,
    chess.KING: 11,
}

_CHESS_FEAT_DIM = 19


def encode_chess_position(board: chess.Board) -> np.ndarray:
    """Encode a chess position into a [64, 19] float32 tensor.

    Encoding is deterministic and bit-exact with the C++ counterpart.

    Args:
        board: A ``chess.Board`` at any legal or quasi-legal position.

    Returns:
        ``np.ndarray`` of shape ``[64, 19]``, dtype ``float32``.
        Square ordering: a1=0, …, h8=63.
    """
    out = np.zeros((64, _CHESS_FEAT_DIM), dtype=np.float32)

    # --- Planes 0-11: piece placement ---
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        if piece.color == chess.WHITE:
            out[sq, _WHITE_PIECE_PLANES[piece.piece_type]] = 1.0
        else:
            out[sq, _BLACK_PIECE_PLANES[piece.piece_type]] = 1.0

    # --- Plane 12: side-to-move (broadcast) ---
    if board.turn == chess.WHITE:
        out[:, 12] = 1.0

    # --- Planes 13-16: castling rights (broadcast) ---
    if board.has_kingside_castling_rights(chess.WHITE):
        out[:, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        out[:, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        out[:, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        out[:, 16] = 1.0

    # --- Plane 17: en-passant target square ---
    if board.ep_square is not None:
        out[board.ep_square, 17] = 1.0

    # --- Plane 18: half-move clock / 100 (broadcast) ---
    out[:, 18] = board.halfmove_clock / 100.0

    return out


# ---------------------------------------------------------------------------
# Chess move ↔ policy index (LC0 1858-slot compact schema)
# ---------------------------------------------------------------------------
# We implement the full 4672-slot (64 × 73) encoding and map it to the 1858
# compact table.  The compact table is built once at module load time.
#
# Move types per from-square (73 total):
#   0-55:  queen-like rays — (direction, distance)
#          direction order: N, NE, E, SE, S, SW, W, NW  (8 dirs × 7 distances)
#   56-63: knight jumps — standard 8 knight deltas
#   64-72: underpromotions — 3 dirs × 3 pieces (N/B/R; queen-promo = ray slot)

# Direction deltas (file_delta, rank_delta) for queen-like rays
_RAY_DIRS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# Knight jump deltas (file_delta, rank_delta)
_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Underpromotion directions: relative file movement (-1 = left, 0 = straight, +1 = right)
# Each paired with queen-like N/NE/NW ray to identify the "from" direction
_UNDERPROM_DIRS = [-1, 0, 1]  # file_delta (rank delta is always +1 for white)
_UNDERPROM_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _sq(file: int, rank: int) -> int:
    """Convert (file, rank) 0-based to square index 0-63."""
    return rank * 8 + file


def _build_chess_policy_maps() -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]]]:
    """Build the (from_sq, move_type) ↔ compact index 1858 mapping.

    Returns:
        forward: (from_sq, move_type_73) → compact_idx
        inverse: compact_idx → (from_sq, move_type_73)
    """
    forward: dict[tuple[int, int], int] = {}
    inverse: dict[int, tuple[int, int]] = {}
    compact_idx = 0

    for from_sq in range(64):
        from_file = from_sq % 8
        from_rank = from_sq // 8

        # Queen-like rays (move types 0-55)
        for dir_idx, (df, dr) in enumerate(_RAY_DIRS):
            for dist in range(1, 8):  # distances 1-7
                to_file = from_file + df * dist
                to_rank = from_rank + dr * dist
                if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                    break  # off-board; further distances also off-board in this direction
                move_type = dir_idx * 7 + (dist - 1)  # 0-55
                key = (from_sq, move_type)
                if key not in forward:
                    forward[key] = compact_idx
                    inverse[compact_idx] = key
                    compact_idx += 1

        # Knight jumps (move types 56-63)
        for k_idx, (df, dr) in enumerate(_KNIGHT_DELTAS):
            to_file = from_file + df
            to_rank = from_rank + dr
            if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                continue  # off-board
            move_type = 56 + k_idx
            key = (from_sq, move_type)
            if key not in forward:
                forward[key] = compact_idx
                inverse[compact_idx] = key
                compact_idx += 1

        # Underpromotions (move types 64-72) — only from rank 6 (7th rank, 0-indexed)
        if from_rank == 6:  # white underpromotes from rank 7 (index 6)
            for dir_j, df in enumerate(_UNDERPROM_DIRS):
                for piece_k, _ in enumerate(_UNDERPROM_PIECES):
                    to_file = from_file + df
                    to_rank = from_rank + 1  # always forward
                    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                        continue
                    move_type = 64 + dir_j * 3 + piece_k
                    key = (from_sq, move_type)
                    if key not in forward:
                        forward[key] = compact_idx
                        inverse[compact_idx] = key
                        compact_idx += 1

    return forward, inverse


# Build once at module load
_CHESS_POLICY_FWD, _CHESS_POLICY_INV = _build_chess_policy_maps()
_CHESS_NUM_COMPACT = len(_CHESS_POLICY_FWD)  # should be 1858


def _chess_move_to_move_type(move: chess.Move, board: chess.Board) -> tuple[int, int]:
    """Return (from_sq, move_type_73) for a legal chess move."""
    from_sq = move.from_square
    to_sq = move.to_square
    from_file = from_sq % 8
    from_rank = from_sq // 8
    to_file = to_sq % 8
    to_rank = to_sq // 8
    df = to_file - from_file
    dr = to_rank - from_rank

    # Determine if this is a knight move
    piece = board.piece_at(from_sq)
    is_knight = piece is not None and piece.piece_type == chess.KNIGHT

    if is_knight:
        delta = (df, dr)
        k_idx = _KNIGHT_DELTAS.index(delta)
        return (from_sq, 56 + k_idx)

    # Underpromote? (promotion to N/B/R)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # It's an underpromote
        dir_j = df + 1  # df in {-1,0,+1} → 0,1,2
        piece_k = _UNDERPROM_PIECES.index(move.promotion)
        return (from_sq, 64 + dir_j * 3 + piece_k)

    # Queen-like ray (including queen promotions)
    if df == 0:
        dir_idx = 0 if dr > 0 else 4  # N or S
    elif dr == 0:
        dir_idx = 2 if df > 0 else 6  # E or W
    else:
        # Diagonal
        if df > 0 and dr > 0:
            dir_idx = 1  # NE
        elif df > 0 and dr < 0:
            dir_idx = 3  # SE
        elif df < 0 and dr < 0:
            dir_idx = 5  # SW
        else:
            dir_idx = 7  # NW

    dist = max(abs(df), abs(dr))
    return (from_sq, dir_idx * 7 + (dist - 1))


def chess_move_to_idx(move: chess.Move, board: chess.Board) -> int:
    """Map a legal chess move to its compact policy index.

    Returns an integer in ``[0, 1857]`` when the move is in the compact map.
    Returns ``-1`` when the move falls outside the 1858-slot schema (rare
    underpromotion directions that LC0's compact map excludes). The caller
    should skip such positions when generating training data.
    """
    key = _chess_move_to_move_type(move, board)
    return _CHESS_POLICY_FWD.get(key, -1)


def chess_idx_to_move(idx: int, board: chess.Board) -> chess.Move | None:
    """Map a compact policy index back to a chess.Move.

    Returns ``None`` if the index has no corresponding legal move on ``board``.

    Args:
        idx:   Compact index in ``[0, 1857]``.
        board: Current board position.

    Returns:
        A ``chess.Move`` or ``None``.
    """
    if idx not in _CHESS_POLICY_INV:
        return None
    from_sq, move_type = _CHESS_POLICY_INV[idx]
    from_file = from_sq % 8
    from_rank = from_sq // 8

    if move_type < 56:
        # Queen-like ray
        dir_idx = move_type // 7
        dist = (move_type % 7) + 1
        df, dr = _RAY_DIRS[dir_idx]
        to_file = from_file + df * dist
        to_rank = from_rank + dr * dist
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            return None
        to_sq = _sq(to_file, to_rank)
        # Check for queen promotion
        promotion = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank == 7:
            promotion = chess.QUEEN
        elif piece is not None and piece.piece_type == chess.PAWN and to_rank == 0:
            promotion = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promotion)

    elif move_type < 64:
        # Knight jump
        k_idx = move_type - 56
        df, dr = _KNIGHT_DELTAS[k_idx]
        to_file = from_file + df
        to_rank = from_rank + dr
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            return None
        return chess.Move(from_sq, _sq(to_file, to_rank))

    else:
        # Underpromote
        under_idx = move_type - 64
        dir_j = under_idx // 3
        piece_k = under_idx % 3
        df = dir_j - 1  # {-1, 0, +1}
        to_file = from_file + df
        # Determine direction of promotion (white goes up, black goes down)
        piece = board.piece_at(from_sq)
        if piece is not None and piece.color == chess.WHITE:
            to_rank = from_rank + 1
        else:
            to_rank = from_rank - 1
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            return None
        promo_piece = _UNDERPROM_PIECES[piece_k]
        return chess.Move(from_sq, _sq(to_file, to_rank), promotion=promo_piece)


# ---------------------------------------------------------------------------
# Shogi encoding (§3) — uses python-shogi
# ---------------------------------------------------------------------------
# python-shogi is a pure-Python shogi library with SFEN support.
# It is imported lazily so that chess-only code paths don't require it.

_SHOGI_FEAT_DIM = 90
_SHOGI_SEQ_LEN = 81

# Piece type indices for own-side planes (planes 0-13)
# Following dlshogi ordering: 歩 香 桂 銀 角 飛 金 玉 と 成香 成桂 成銀 馬 龍
# In python-shogi: PAWN=1, LANCE=2, KNIGHT=3, SILVER=4, GOLD=5, BISHOP=6,
#   ROOK=7, KING=8, PROM_PAWN=9, PROM_LANCE=10, PROM_KNIGHT=11,
#   PROM_SILVER=12, PROM_BISHOP=13, PROM_ROOK=14
_SHOGI_PIECE_PLANE_MAP: dict[int, int] = {
    1: 0,   # 歩 PAWN
    2: 1,   # 香 LANCE
    3: 2,   # 桂 KNIGHT
    4: 3,   # 銀 SILVER
    6: 4,   # 角 BISHOP
    7: 5,   # 飛 ROOK
    5: 6,   # 金 GOLD
    8: 7,   # 玉 KING
    9: 8,   # と PROM_PAWN
    10: 9,  # 成香 PROM_LANCE
    11: 10, # 成桂 PROM_KNIGHT
    12: 11, # 成銀 PROM_SILVER
    13: 12, # 馬 PROM_BISHOP
    14: 13, # 龍 PROM_ROOK
}

# Hand piece types and their plane indices within the 7-plane hand block
# Order: 歩/香/桂/銀/金/角/飛
# python-shogi: PAWN=1, LANCE=2, KNIGHT=3, SILVER=4, GOLD=5, BISHOP=6, ROOK=7
_SHOGI_HAND_TYPES = [1, 2, 3, 4, 5, 6, 7]  # PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK
# Maximum number of each piece in hand (used for normalisation)
_SHOGI_HAND_MAX = {
    1: 18,  # 歩 (18 pawns)
    2: 4,   # 香
    3: 4,   # 桂
    4: 4,   # 銀
    5: 4,   # 金
    6: 2,   # 角
    7: 2,   # 飛
}


def _try_import_shogi():  # type: ignore[return]
    """Import python-shogi, raising a clear error if unavailable."""
    try:
        import shogi  # type: ignore[import]
        return shogi
    except ImportError as exc:
        raise ImportError(
            "python-shogi is required for shogi encoding.  "
            "Install it with:  uv add python-shogi"
        ) from exc


def _shogi_square_to_idx(sq_int: int) -> int:
    """Convert python-shogi square integer to cshogi-convention index.

    python-shogi uses: file 9 = 0, file 8 = 1, …, file 1 = 8 per rank,
    then rank 1 = 0..8, rank 2 = 9..17, … rank 9 = 72..80.
    Actually python-shogi squares are 0-80 with square = (9-file)*9+(rank-1).
    That already matches the cshogi convention in encoding_spec §3.1.
    """
    return sq_int


def encode_shogi_position(sfen_or_board: "str | object") -> np.ndarray:
    """Encode a shogi position into a [81, 90] float32 tensor.

    Follows encoding_spec.md §3 (v1 simplified — planes 28-55, 70-88 zero).
    Side-to-move handling: if 後手 is to move, board is rotated 180° and
    colours are swapped before encoding (dlshogi convention).

    Args:
        sfen_or_board: Either an SFEN string or a python-shogi ``Board`` object.

    Returns:
        ``np.ndarray`` of shape ``[81, 90]``, dtype ``float32``.

    Raises:
        ImportError: If python-shogi is not installed.
        NotImplementedError: Raised for unsupported inputs.
    """
    shogi = _try_import_shogi()

    if isinstance(sfen_or_board, str):
        board = shogi.Board(sfen_or_board)
    else:
        board = sfen_or_board

    out = np.zeros((_SHOGI_SEQ_LEN, _SHOGI_FEAT_DIM), dtype=np.float32)

    # Determine if we need to mirror for 後手-to-move
    mirror = (board.turn == shogi.WHITE)  # WHITE = 後手 (second player)

    # ---------------------------------------------------------------
    # Planes 0-13: own-side pieces; 14-27: opponent-side pieces
    # ---------------------------------------------------------------
    for sq in range(81):
        piece = board.piece_at(sq)
        if piece is None:
            continue

        piece_type = piece.piece_type
        piece_color = piece.color  # BLACK=0=先手, WHITE=1=後手

        if mirror:
            # Mirror: own side becomes WHITE (後手), so we flip colour interpretation
            # Also flip square: mirrored_sq = 80 - sq
            enc_sq = 80 - sq
            is_own = (piece_color == shogi.WHITE)
        else:
            enc_sq = sq
            is_own = (piece_color == shogi.BLACK)

        plane_offset = 0 if is_own else 14
        plane = plane_offset + _SHOGI_PIECE_PLANE_MAP.get(piece_type, -1)
        if plane >= 0:
            out[enc_sq, plane] = 1.0

    # ---------------------------------------------------------------
    # Planes 56-62: own-side hand pieces (broadcast, normalised)
    # Planes 63-69: opponent-side hand pieces (broadcast, normalised)
    # ---------------------------------------------------------------
    if mirror:
        own_color = shogi.WHITE
        opp_color = shogi.BLACK
    else:
        own_color = shogi.BLACK
        opp_color = shogi.WHITE

    # pieces_in_hand is a list of two Counter objects: [BLACK_hand, WHITE_hand]
    own_hand = board.pieces_in_hand[own_color]
    opp_hand = board.pieces_in_hand[opp_color]

    for hand_plane, piece_type in enumerate(_SHOGI_HAND_TYPES):
        # Own hand
        own_count = own_hand.get(piece_type, 0)
        own_max = _SHOGI_HAND_MAX[piece_type]
        out[:, 56 + hand_plane] = own_count / own_max

        # Opponent hand
        opp_count = opp_hand.get(piece_type, 0)
        opp_max = _SHOGI_HAND_MAX[piece_type]
        out[:, 63 + hand_plane] = opp_count / opp_max

    # ---------------------------------------------------------------
    # Plane 89: side-to-move (1 = 先手, 0 = 後手) — broadcast
    # ---------------------------------------------------------------
    # After mirroring, own side is always "the mover", but we still record
    # the original side-to-move in the global frame.
    if board.turn == shogi.BLACK:  # 先手
        out[:, 89] = 1.0

    # Planes 28-55, 70-88: zero-filled in v1 (see encoding_spec §3.2 note)

    return out


# ---------------------------------------------------------------------------
# Shogi move index helpers
# ---------------------------------------------------------------------------

def shogi_move_to_idx(move: "object", board: "object") -> int:
    """Map a python-shogi move to dlshogi compact policy index (0-2186).

    dlshogi encoding: 81 squares × 27 move types = 2187
    Move types 0-19: 10 directions × 2 (no-promote / promote)
    Move types 20-26: drop types (7 hand pieces)

    Args:
        move:  A python-shogi ``Move`` object.
        board: Current python-shogi ``Board`` (unused currently, kept for API).

    Returns:
        Integer in ``[0, 2186]``.
    """
    shogi = _try_import_shogi()

    _SHOGI_DIR_MAP = {
        # (file_delta, rank_delta) → direction_index (0-9)
        # File increases right, rank increases down in shogi (from black's POV)
        (0, -1): 0,   # Forward (up for 先手)
        (1, -1): 1,   # Forward-right
        (-1, -1): 2,  # Forward-left
        (0, 1): 3,    # Backward
        (1, 1): 4,    # Backward-right
        (-1, 1): 5,   # Backward-left
        (1, 0): 6,    # Right
        (-1, 0): 7,   # Left
        # Knight jumps
        (1, -2): 8,   # Knight forward-right
        (-1, -2): 9,  # Knight forward-left
    }

    if move.drop_piece_type:
        # Drop move
        drop_type_idx = _SHOGI_HAND_TYPES.index(move.drop_piece_type)
        move_type = 20 + drop_type_idx
        return move.to_square * 27 + move_type

    # Board move
    from_sq = move.from_square
    to_sq = move.to_square

    from_file = 8 - (from_sq % 9)  # convert back to 1-9
    from_rank = from_sq // 9 + 1
    to_file = 8 - (to_sq % 9)
    to_rank = to_sq // 9 + 1

    df = to_file - from_file
    dr = to_rank - from_rank

    # Normalise to unit direction for sliding pieces
    gcd_val = max(abs(df), abs(dr))
    if gcd_val > 0:
        unit_df = df // gcd_val if df != 0 and abs(df) == gcd_val else (1 if df > 0 else (-1 if df < 0 else 0))
        unit_dr = dr // gcd_val if dr != 0 and abs(dr) == gcd_val else (1 if dr > 0 else (-1 if dr < 0 else 0))
        # Actually for knights the delta is not normalised
        if abs(df) == 1 and abs(dr) == 2:
            unit_df, unit_dr = df, dr
        elif abs(df) == 2 and abs(dr) == 1:
            # not a valid shogi knight
            unit_df, unit_dr = df // 2, dr  # shouldn't happen
        else:
            unit_df = 0 if df == 0 else (1 if df > 0 else -1)
            unit_dr = 0 if dr == 0 else (1 if dr > 0 else -1)
    else:
        unit_df, unit_dr = 0, 0

    dir_idx = _SHOGI_DIR_MAP.get((unit_df, unit_dr), 0)
    promote_offset = 10 if move.promotion else 0
    move_type = dir_idx + promote_offset
    return from_sq * 27 + move_type


def shogi_idx_to_move(idx: int, board: "object") -> "object | None":
    """Map a dlshogi policy index back to a python-shogi Move.

    Returns ``None`` if the index is out of range.
    """
    shogi = _try_import_shogi()

    if not (0 <= idx < 2187):
        return None

    from_sq = idx // 27
    move_type = idx % 27

    if move_type >= 20:
        # Drop move
        piece_type = _SHOGI_HAND_TYPES[move_type - 20]
        return shogi.Move(0, from_sq, False, piece_type)  # from=0 for drops

    promoted = move_type >= 10
    dir_idx = move_type % 10

    _SHOGI_IDX_TO_DIR = [
        (0, -1), (1, -1), (-1, -1),
        (0, 1), (1, 1), (-1, 1),
        (1, 0), (-1, 0),
        (1, -2), (-1, -2),
    ]
    if dir_idx >= len(_SHOGI_IDX_TO_DIR):
        return None

    df, dr = _SHOGI_IDX_TO_DIR[dir_idx]
    from_file_1indexed = 8 - (from_sq % 9)
    from_rank_1indexed = from_sq // 9 + 1
    to_file = from_file_1indexed + df
    to_rank = from_rank_1indexed + dr

    if not (1 <= to_file <= 9 and 1 <= to_rank <= 9):
        return None

    to_sq = (to_rank - 1) * 9 + (8 - to_file)
    return shogi.Move(from_sq, to_sq, promoted)
