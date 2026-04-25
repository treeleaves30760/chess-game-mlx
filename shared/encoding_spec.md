# NN Input / Output Encoding Specification

This document is the **authoritative contract** between:
- `engine/include/nn/input_encoder.hpp` (C++)
- `training/src/training/data/encoding.py` (Python)

Both sides must produce **bit-exact identical tensors** for any given position. A golden-data test suite in `shared/golden_data/` enforces this every CI run.

Any change here requires updating both implementations and regenerating golden data.

---

## 1. Chess input encoding

### 1.1 Shape
`[64, 19]` float32 tensor, row-major. 64 squares × 19 feature planes.

Square indexing: **a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.** (same as python-chess / chess-library).

### 1.2 Feature planes (the 19 features per square)

| Index | Feature | Value |
|---|---|---|
| 0 | White pawn at this square | {0, 1} |
| 1 | White knight | {0, 1} |
| 2 | White bishop | {0, 1} |
| 3 | White rook | {0, 1} |
| 4 | White queen | {0, 1} |
| 5 | White king | {0, 1} |
| 6 | Black pawn | {0, 1} |
| 7 | Black knight | {0, 1} |
| 8 | Black bishop | {0, 1} |
| 9 | Black rook | {0, 1} |
| 10 | Black queen | {0, 1} |
| 11 | Black king | {0, 1} |
| 12 | Side-to-move is white | {0, 1} (broadcast to all 64 squares) |
| 13 | White king-side castling right | {0, 1} (broadcast) |
| 14 | White queen-side castling right | {0, 1} (broadcast) |
| 15 | Black king-side castling right | {0, 1} (broadcast) |
| 16 | Black queen-side castling right | {0, 1} (broadcast) |
| 17 | En passant target square | {0, 1} (only the EP square is 1) |
| 18 | Half-move clock / 100.0 | [0, 1] (broadcast) |

**Rationale**: This is a flat 19-channel per-square encoding, sufficient for a transformer with 64 tokens. For a CNN alternative, reshape to `[19, 8, 8]`.

**Side-to-move normalisation**: We do **NOT** mirror the board when black is to move. The model learns asymmetry through feature 12. Policy head outputs are always in white-oriented move coordinates; for black moves, the model learns to flip implicitly via feature 12. This avoids subtle bugs from inconsistent mirroring.

---

## 2. Chess policy head encoding

Following the AlphaZero convention (used by LC0) — **4672 total move indices**, but we use the compact **1858-move schema** from LC0 which skips invalid combinations.

### 2.1 Layout
Each move is encoded as `(from_square, move_type)` where `move_type` is one of:
- 56 queen-like rays (8 directions × 7 distances)
- 8 knight jumps
- 9 underpromotions (3 directions × 3 promotion pieces: knight, bishop, rook; queen promotion uses the queen-ray slot)

Total move types per from-square: `56 + 8 + 9 = 73`.
Full space: `64 × 73 = 4672`. Many are always illegal (e.g. knight-move from corner to off-board), so we use LC0's **policy_map.txt** compressed mapping (1858 slots). The mapping tables live at:
- `shared/policy_map_chess.txt` (generated, checked in)

### 2.2 API

**C++** (`engine/include/chess/chess_encoding.hpp`):
```cpp
int move_to_policy_idx(const ChessMove& m, const ChessPosition& pos) noexcept;
ChessMove policy_idx_to_move(int idx, const ChessPosition& pos) noexcept;
```

**Python** (`training/src/training/data/encoding.py`):
```python
def chess_move_to_idx(move: chess.Move, board: chess.Board) -> int: ...
def chess_idx_to_move(idx: int, board: chess.Board) -> chess.Move: ...
```

### 2.3 Value head
Scalar, range `[-1, +1]` via `tanh`. **Sign convention: positive = white/先手 advantage.** This is consistent regardless of side-to-move (i.e. we do NOT flip the sign when black is to move).

### 2.4 Moves-left head (auxiliary)
Scalar, non-negative, predicting the number of half-moves until game end. Predicted via `softplus(raw_output)`. Training target: actual half-moves remaining in the game record.

---

## 3. Shogi input encoding

### 3.1 Shape
`[81, C]` float32 tensor, where `C = 90`. Square count is `9 × 9 = 81`.

Square indexing: follows **cshogi convention** — 先手視点、file 9 (leftmost from black's perspective) + rank 1 = 0, ..., file 1 + rank 9 = 80. Specifically:
```
square_index = (9 - file) * 9 + (rank - 1)
```
where `file` ∈ {1..9} (right-to-left from black's POV) and `rank` ∈ {1..9} (top-to-bottom from black's POV).

This matches `cshogi.SQUARES` ordering.

### 3.2 Feature planes (90 channels per square)

Planes follow dlshogi's standard feature layout (which is the SOTA shogi encoding as of 2024):

| Range | Count | Feature |
|---|---|---|
| 0..13 | 14 | Own-side pieces, one-hot per piece type: 歩 香 桂 銀 角 飛 金 玉 と 成香 成桂 成銀 馬 龍 |
| 14..27 | 14 | Opponent-side pieces, same 14 piece types |
| 28..41 | 14 | Own-side piece attack counts (each plane = 1 if ≥k attackers, saturated at 3) — **simplified in v1 to 14 binary planes** |
| 42..55 | 14 | Opponent-side piece attack counts — **simplified in v1** |
| 56..62 | 7  | Own hand pieces (count, normalized by max): 歩/香/桂/銀/金/角/飛 (broadcast to all 81 squares) |
| 63..69 | 7  | Opponent hand pieces (broadcast) |
| 70..83 | 14 | Move history last 1 ply (piece placements) — **v1: zero-filled, v2: wire up** |
| 84..88 | 5  | Reserved (king-safety features) — **v1: zero-filled** |
| 89 | 1 | Side-to-move (broadcast; 1 = 先手 to move, 0 = 後手 to move) |

**Note on v1 simplifications**: We start with piece placement + hand + side-to-move (indices 0..27, 56..69, 89), leaving 28..55 and 70..88 as zero-filled. This is 43 active channels out of 90. Easy to upgrade later without changing the tensor shape (which is the expensive part to change).

### 3.3 Side-to-move handling
Unlike chess, dlshogi mirrors the board so that the **side-to-move is always "own side"**. This means:
- If 先手 (black) to move: board is as-is
- If 後手 (white) to move: board is rotated 180° and pieces are colour-swapped

This simplifies the model — policy head always predicts **"our" moves**. Feature 89 becomes a disambiguator for asymmetric aspects (none in shogi, so it's mostly informational).

Both C++ and Python encoders must apply this mirroring identically.

### 3.4 Shogi policy head
2187 total move indices, using dlshogi's move encoding:
- 81 squares × 27 "move types" = 2187
- Move types include: 10 directions (8 compass + 2 knight) × 2 (promote / no-promote) = 20 board moves + 7 drop types = 27

Full mapping in `shared/policy_map_shogi.txt` (generated, checked in).

### 3.5 Value head
Same as chess: tanh, [-1, +1], positive = 先手 advantage, **not flipped by side-to-move**. (Note: because of §3.3 mirroring, the model sees the flipped board for 後手 moves, but the value sign is in the original un-mirrored frame.)

---

## 4. Batch layout

Both chess and shogi use a flat batch:
- Chess batch: `[B, 64, 19]`
- Shogi batch: `[B, 81, 90]`

Data type: `bfloat16` during training on M3 (native bf16 support). `float32` during C++ encoding; conversion happens at the MLX boundary.

---

## 5. Golden data tests

`shared/golden_data/` contains:
- `chess_positions.json` — 100 FEN strings covering varied positions (opening / middle / endgame / promotions / en-passant / castling / stalemate)
- `chess_encodings.npz` — encoded tensors (float32) for each of the 100 positions
- `shogi_positions.json` — 100 SFEN strings (mirror coverage for shogi)
- `shogi_encodings.npz` — encoded tensors

C++ test (`engine/tests/test_encoding_chess.cpp`) loads the JSON, encodes each position, compares to the `.npz` reference. Equality must be **bit-exact** for float32 (since both paths do pure integer/boolean placements — no FP arithmetic).

Python test (`training/tests/test_encoding.py`) does the reverse: load FEN, encode via Python, compare to the same `.npz`.

Regenerating golden data (when encoding spec changes):
```bash
uv run python -m training.scripts.regen_golden_data
```

---

## 6. Versioning

This spec is tagged `v1.0`. Breaking changes bump the major version and require regenerating:
- `shared/golden_data/`
- All trained model checkpoints (old weights are incompatible)

Current version: **v1.0** (2026-04-23).
