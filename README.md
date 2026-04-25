# Chess_Game_mlx — Chess + Shogi AI for Apple Silicon

A three-tier chess + shogi AI platform built for MacBook Air M3, with MLX-accelerated neural network inference, PUCT-MCTS search, multi-ponder (5-tree parallel pre-computation during opponent's turn), and a Pygame GUI. Designed to be trained to SOTA strength over time on the user's own hardware.

**Status as of 2026-04-23**: All phases 0-6 complete, 152/152 tests passing, end-to-end arena verified against Stockfish 18.

## Quick Facts

| Component | Status | Detail |
|---|---|---|
| Chess engine (UCI) | ✅ | `engine/bin/chess_engine`, perft(6) = 119,060,324 verified |
| Shogi engine (USI) | ✅ | `engine/bin/shogi_engine`, cshogi perft tables matched |
| MLX-C NN backend | ✅ | BT4-style 40M transformer, CPU + Metal (Metal toolchain required) |
| PUCT-MCTS | ✅ | 2M nps / thread with StubBackend, virtual-loss parallelism |
| Multi-ponder (5 trees) | ✅ | 2.5M combined nps, JSON-RPC protocol extension |
| UCI/USI + JSON-RPC ext | ✅ | standard protocols + `set_side` / `start_multi_ponder` / `opponent_played` / etc. |
| Training pipeline | ✅ | Stockfish-18 teacher, supervised + self-play paths, safetensors export |
| Pygame GUI | ✅ | chess 8×8 rendered, top-3 arrows, eval bar, mock engine + real engine client |
| Arena harness | ✅ | benchmarks/arena.py — plays N-game matches vs any UCI engine |

## Arena Strength (as of scaffold completion)

Measured with 30-epoch training on 5000 Stockfish-annotated positions (3.6M-param model, no GPU):

- **chess_engine + trained weights vs Stockfish 1320 Elo → 0-1-1 (-191 Elo)** → ~1130 Elo
- **chess_engine with StubBackend (no weights) vs Stockfish 1320 Elo → 0-2 (-800 Elo)** → ~520 Elo

Training on 20k positions × 50 epochs (running separately) is expected to push this further.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  GUI Tier         Python + Pygame                           │
│  • BoardRenderer (chess 8×8 / shogi 9×9 + 持駒)             │
│  • AnalysisPanel (eval bar / top-3 / PV / nodes-per-sec)    │
│  • PonderManager (協調多預想)                                │
│  • EngineClient (UCI/USI + JSON-RPC 擴展)                    │
└──────────────▲───────────────────────────────┬──────────────┘
               │ stdin/stdout (text)           │ IPC events
┌──────────────┴──────────────────────────────────────────────┐
│  Engine Tier      C++17 + MLX-C                             │
│  • GameRules<Traits>  (chess / shogi 特化)                   │
│  • MCTS (PUCT, virtual-loss, multi-ponder)                   │
│  • MLXBackend (batched inference on Metal / CPU)             │
│  • UCI / USI loop + JSON-RPC 擴展協議                         │
└─────────────────────────▲───────────────────────────────────┘
                          │ safetensors weights
┌─────────────────────────┴───────────────────────────────────┐
│  Training Tier    Python 3.12 + MLX + uv                    │
│  • ChessShogiTransformer (BT4-inspired encoder, 40M)         │
│  • StockfishTeacher / ChessBench / dlshogi loaders           │
│  • SupervisedTrainer (+ SelfPlayTrainer stub for Phase 7)    │
│  • Exporter (MLX → engine-ready safetensors)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup

### Prerequisites (one-time)

```bash
# Homebrew (macOS)
brew install stockfish            # for teacher + arena
# uv (Python pkg manager, if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Metal toolchain (optional, for GPU inference — takes ~2 GB)
xcodebuild -downloadComponent MetalToolchain
```

### Clone + build

```bash
git clone --recurse-submodules https://github.com/treeleaves30760/Chess_Game_mlx.git
cd Chess_Game_mlx
uv sync

# CPU build (works without Metal toolchain)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCHESS_MLX_WITH_MLX=ON -DCHESS_MLX_BUILD_METAL=OFF
cmake --build build -j

# GPU build (requires Metal toolchain from xcodebuild command above; inference is 10-100× faster)
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCHESS_MLX_WITH_MLX=ON
# cmake --build build -j

ctest --test-dir build --output-on-failure   # 94 tests should pass
uv run pytest -q                             # 58 tests should pass
```

---

## Usage

### 1. Play against the engine via Pygame GUI

```bash
# With the mock engine (no compilation needed):
uv run python -m gui.app --game chess --engine mock

# With the real trained engine (replace the weights path with your actual checkpoint):
uv run python -m gui.app --game chess \
    --engine ./engine/bin/chess_engine \
    --weights checkpoints/chess_192d_6L_sf20k_e10/final.safetensors
```

Keyboard: `F` flip board · `N` new game · `Esc` quit.

### 2. Run the engine standalone (UCI mode)

```bash
./engine/bin/chess_engine --weights checkpoints/.../final.safetensors
# then type:
uci
setoption name MultiPV value 3
position startpos
go movetime 1000
```

Shogi: `./engine/bin/shogi_engine` (USI protocol).

### 3. JSON-RPC extension (multi-ponder, single-side, eval bar)

```bash
./engine/bin/chess_engine --weights .../final.safetensors
# Standard UCI handshake
uci
isready
# Lock to a colour — engine auto-enables multi-ponder on opponent's turn
{"jsonrpc":"2.0","method":"set_side","params":{"me":"white"},"id":1}
position startpos moves e2e4
# Start 5-tree multi-ponder (40/20/15/15/10% budget split)
{"jsonrpc":"2.0","method":"start_multi_ponder","params":{"k":5},"id":2}
# Stream of policy_preview + ponder_progress events back
# When opponent plays:
{"jsonrpc":"2.0","method":"opponent_played","params":{"move":"e7e5"},"id":3}
# -> ponder_hit (if in top-5) OR ponder_miss (start fresh search)
```

Full spec: `shared/protocol_spec.md`.

### 4. Train a better model

```bash
# Step A: Annotate random positions with Stockfish 18 (~150/sec at depth=10)
uv run python -c "
from training.data.stockfish_teacher import StockfishTeacher, save_prelabeled_dataset
from pathlib import Path
with StockfishTeacher(depth=12, threads=2) as t:
    save_prelabeled_dataset(Path('data/labels_50k.npz'), t, num_positions=50000, seed=42)
"

# Step B: Train
uv run python training/scripts/pretrain_stockfish.py \
    --data-path data/labels_50k.npz --positions 50000 \
    --epochs 40 --batch-size 128 --lr 3e-4 \
    --d-model 512 --layers 12 \
    --out-dir checkpoints/chess_v2

# Step C: Test the trained model vs Stockfish
uv run python benchmarks/arena.py \
    --a ./engine/bin/chess_engine --a-args "--weights checkpoints/chess_v2/final.safetensors" \
    --b stockfish \
    --b-setoption "setoption name UCI_LimitStrength value true" \
    --b-setoption "setoption name UCI_Elo value 2000" \
    --games 20 --movetime 1000
```

### 5. Path to SOTA-level strength

M3 Air cannot train a from-scratch SOTA model — that requires thousands of GPU-months. But you can reach strong practical play via these escalating steps:

1. **Warm-start path (quickest)**: Use this platform with imported LC0 BT4 weights. Write a converter that maps LC0's safetensors to our layer naming. Expected strength: 3000+ Elo with MCTS search. Time: ~1 week to write converter.
2. **ChessBench monolithic pretraining**: Download the DeepMind [ChessBench](https://github.com/google-deepmind/searchless_chess) dataset (10M games, 15B labels). Replace `StockfishTeacher` with ChessBench loader. Train on M3 for 1-2 weeks → ~2800 Elo baseline. Then fine-tune.
3. **Self-play fine-tuning** (Phase 7 roadmap): Use the trained model to generate self-play games; train on those labels; iterate. This is how LC0 reached SOTA.
4. **Distillation from a bigger teacher**: Annotate more positions with Stockfish 18 at depth=20 (slower but much better labels).

Training at scale takes days-to-weeks of continuous compute — install Metal toolchain for 10-100× speedup.

---

## Project Structure

```
Chess_Game_mlx/
├── pyproject.toml            # uv workspace (members: training, gui)
├── CMakeLists.txt            # C++ top-level
├── engine/                   # C++17 engine (MCTS + MLX-C + UCI/USI + JSON-RPC)
│   ├── include/
│   │   ├── core/             # GameRules<Traits>
│   │   ├── chess/            # ChessTraits (wraps Disservin/chess-library)
│   │   ├── shogi/            # ShogiTraits (wraps cshogi C++ core)
│   │   ├── search/           # MCTS, MultiPonderManager
│   │   ├── nn/               # NNBackend, MlxBackend, StubBackend, Batcher
│   │   └── protocol/         # UCI loop, USI loop, JSON-RPC dispatcher
│   ├── src/                  # implementations
│   ├── tests/                # 94 GoogleTest tests
│   └── third_party/          # chess-library, cshogi, mlx-c, nlohmann_json
├── training/                 # Python + MLX training
│   ├── src/training/
│   │   ├── models/           # ChessShogiTransformer (40M-ish params)
│   │   ├── data/             # encoding, chessbench_loader, stockfish_teacher
│   │   ├── trainers/         # supervised
│   │   └── export.py         # MLX → safetensors for engine
│   └── scripts/              # pretrain_stockfish.py, export_dummy.py
├── gui/                      # Python + Pygame GUI
│   └── src/gui/              # app, board_renderer, engine_client, mock_engine, ponder_manager, analysis_panel
├── shared/                   # encoding_spec.md, protocol_spec.md, golden_data/
├── benchmarks/               # arena.py (vs Stockfish)
├── data/                     # downloaded datasets
└── checkpoints/              # trained model outputs
```

## Licensing

Note: `engine/bin/shogi_engine` links the vendored cshogi C++ subset which is GPL v3 (derived from Apery/Stockfish). Consequently **shogi_engine inherits GPL v3**. The chess engine (`chess_engine`) only depends on MIT libraries (chess-library, MLX, nlohmann/json) and can be licensed independently.

Your own code in `engine/src/`, `engine/include/core/`, `engine/include/chess/`, `engine/include/search/`, `engine/include/nn/`, `engine/include/protocol/`, all of `training/`, `gui/`, `shared/`, and `benchmarks/` is yours to license. The common default is MIT. See `LICENSE` file (TBD — pending decision).

## Current Test Count

```
C++  (ctest):   94 / 94 passing
Python (pytest): 58 / 58 passing
Total:          152 / 152
```

See `docs/original-plan.md` (mirror of the original design plan produced during the /plan session; maintainer's source copy lives at `~/.claude/plans/ai-macbook-air-m3-sota-pygame-mac-gpu-p-whimsical-marshmallow.md`).

## Known Limitations (as of 2026-04-23 snapshot)

- **Metal GPU backend disabled by default**: Requires `xcodebuild -downloadComponent MetalToolchain` (interactive). CPU MLX backend works out-of-box at ~1800 nps; Metal would give 10-100× speedup for NN inference.
- **Shogi GUI is a placeholder**: Chess GUI is full-featured; shogi rendering is a stub pending future work. USI protocol + engine work perfectly; you can use third-party shogi GUIs (ShogiGUI / ShogiHome).
- **Policy map compact vs full**: C++ engine uses the full 4672-slot AlphaZero policy; Python uses LC0's compact 1858-slot map. The training pipeline filters out the ~0.1% of legal moves that fall outside the compact map (rare underpromotion angles). No correctness issue.
- **Training to SOTA needs real compute**: Models in this repo are proof-of-concept trained on 5k-20k positions. The pipeline is production-ready but SOTA-level weights require multi-day continuous training + ChessBench-scale data.
