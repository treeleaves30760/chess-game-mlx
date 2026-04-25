# MacBook Air M3 Chess + Shogi AI（MLX 版）實作計劃

## Context

**目標**：在 MacBook Air M3（10-core GPU / 24 GB 統一記憶體）上訓練西洋棋與將棋 AI，目標棋力進入 SOTA 等級。

**現實校準**：
- 真正的 SOTA（LC0 / Stockfish 17 / YaneuraOu 最新版，3600+ Elo）是**上千張 GPU 月級別**訓出來的，M3 Air 無法從零達到。
- 但 M3 Air 24 GB 統一記憶體 + MLX 的 bf16 原生支援完全足以：
  1. 把 **ChessBench（DeepMind 2024 NeurIPS）** 資料集做監督式預訓練出 ~2800 Elo 強度的 40-80M Transformer（參考實證：270M 版本 Lichess blitz 2895 Elo）。
  2. 接 MCTS 搜索再加 200-400 Elo，實戰對弈可上探 3000+ Elo、足以擊敗大多數商用引擎的預設強度。
  3. 將棋以 dlshogi 訓練資料（Stockfish-等級的 YaneuraOu 生成）走同路線。
- **結論**：可行的「beat-SOTA-competitor」定位是「在 M3 Air 上跑的對局引擎比絕大多數商用/家用引擎強」，不是「從零訓出 LC0 級別網路」。用戶已於問卷中選擇「ChessBench 監督式預訓練 + 自我對弈微調」路線。

**用戶決策（已確認）**：
1. 西洋棋 + 將棋**平行開發**（共用抽象層）
2. 引擎語言：**C++ + MLX-C**（配合 pybind11 給訓練端）
3. 訓練策略：**ChessBench 監督式預訓練 + 自我對弈微調**
4. 硬體：**M3 Air 10-core GPU / 24 GB**
5. 引擎基礎：**自寫 MCTS + 借用成熟 move-gen**（chess-library + cshogi C++ core）

---

## 1. 整體架構（三層）

```
┌─────────────────────────────────────────────────────────────┐
│  GUI Tier         Python + Pygame                           │
│  • BoardRenderer (chess 8×8 / shogi 9×9 + 持駒)             │
│  • AnalysisPanel (eval bar / top-3 / PV / nodes-per-sec)    │
│  • PonderManager (協調多預想)                                │
│  • EngineClient (UCI/USI + JSON-RPC 擴展)                    │
└──────────────▲───────────────────────────────┬──────────────┘
               │ stdin/stdout (text)           │ IPC events
               │                               ▼
┌──────────────┴──────────────────────────────────────────────┐
│  Engine Tier      C++17 + MLX-C                             │
│  • GameRules<Traits> (chess / shogi 特化)                    │
│  • MCTS (PUCT, virtual-loss, multi-ponder)                   │
│  • MLXBackend (batched inference on Metal via unified mem)   │
│  • UCI / USI loop + JSON-RPC 擴展協議                         │
│  • move-gen 借：Disservin/chess-library + cshogi C++ core    │
└─────────────────────────▲───────────────────────────────────┘
                          │ safetensors weights
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  Training Tier    Python 3.12 + MLX + uv                    │
│  • ChessShogiTransformer (BT4-inspired encoder, 40-60M)      │
│  • ChessBenchLoader / dlshogiLoader                          │
│  • SupervisedTrainer / SelfPlayTrainer / Distiller           │
│  • Exporter (MLX → engine-ready safetensors)                 │
└─────────────────────────────────────────────────────────────┘
```

**為什麼這個切分在 M3 上特別合理**：MLX 的 unified memory 設計讓 C++ 引擎的 MCTS thread pool、5 個 ponder tree、以及 NN 推論張量**共用同一塊 24 GB 實體記憶體**、零拷貝；Python 訓練端透過 `mx.save_safetensors` 匯出權重、C++ 端用 `mlx-c` 載入同一檔案，避免兩套推論路徑造成的精度漂移。

---

## 2. Model 架構（Stage-1 SOTA-track）

**型號**：`ChessShogiTransformer`（共用 backbone、game-specific input & policy head）

| 項目 | v1 規格 | 備註 |
|---|---|---|
| 架構 | Encoder-only Post-LN Transformer | 參考 LC0 BT4 |
| 參數量 | **~40 M**（16 GB OK）/ **~60 M**（24 GB 推薦） | 24 GB 機型可上到 80 M |
| 深度 / 寬度 | 12 layers / d_model=768 / 12 heads / ffn=3072 | |
| 特色模組 | Smolgen（動態注意力偏置）、trainable positional embed、Mish、DeepNet init | BT4 證實比純 CNN 多 270 Elo |
| Input | Chess: 64 tokens × 19 feats；Shogi: 81 tokens × ~45 feats + hand-piece embedding | 編碼定義於 `shared/encoding_spec.md` |
| Policy Head | Chess 1858-move vocab / Shogi 2187-move vocab（dlshogi 標準） | 各棋類獨立 head |
| Value Head | tanh，[-1, +1]（正為白/先手優勢） | |
| Moves-Left Head | L2 回歸剩餘步數（AlphaZero 輔助 loss） | 穩定訓練 |
| 精度 | bf16 訓練（M3 原生）、int8 可選推論 | |

**訓練資料**：
- 西洋棋：[ChessBench（DeepMind 2024）](https://github.com/google-deepmind/searchless_chess) — 10 M 局、15 B action-value 標註、Stockfish 16 生成。抽樣 500 M–2 B positions 即足（M3 上完跑 15 B 不現實）。
- 將棋：dlshogi 公開訓練資料（YaneuraOu 生成的 teacher positions）。

**訓練時程估計（M3 Air 24 GB、60 M 模型、bf16）**：
- 吞吐約 15–25 K positions/sec
- 500 M positions 預訓練 ≈ 6–10 天不間斷
- Self-play 微調：每小時 ~1 K 高質量 self-play 局（400 nodes/move），持續跑 1-3 個月

---

## 3. Engine 核心設計

### 3.1 GameRules 抽象層（chess/shogi 共用）

**Critical file**：`engine/include/core/game_rules.hpp`

```cpp
template <typename Traits>
struct GameRules {
    using Position = typename Traits::Position;
    using Move     = typename Traits::Move;

    static void  generate_legal(const Position&, MoveList&);
    static void  apply(Position&, Move);
    static void  undo(Position&, Move);
    static bool  is_terminal(const Position&);
    static float terminal_value(const Position&);         // {-1, 0, +1}
    static u64   hash(const Position&);                   // Zobrist
    static void  encode_nn(const Position&, float* out);  // → tensor
    static int   move_to_policy_idx(Move, const Position&);
    static Move  policy_idx_to_move(int, const Position&);
};
```

**實作策略**：
- `engine/include/chess/chess_traits.hpp` → 內部委派 `Disservin/chess-library`（header-only、MIT、已經被 Stockfish Winrate Model 用過、千萬級 perft 驗證過）
- `engine/include/shogi/shogi_traits.hpp` → 內部委派從 `cshogi/src/` 抽出來的 `bitboard.cpp` / `generateMoves.cpp`（C++ 核心、MIT-style、dlshogi 生態圈驗證過）
- 編碼 (`encode_nn`) 與 `move_to_policy_idx` 是兩邊唯一需要自己寫的部分，且**必須與訓練端 Python 編碼 bit-exact 一致**（用黃金測試檔對齊）

### 3.2 MCTS (PUCT)

**Critical file**：`engine/include/search/mcts.hpp`

核心特性：
- PUCT 選擇：`Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))`（LC0 公式）
- Virtual loss：支援 thread pool 並行擴展同一棵樹
- FPU reduction：未訪問子節點的初始 Q 值設為 `parent_Q - k·√P_visited_sum`
- Node 結構 ≤ 64 bytes（cache-line friendly）
- Dirichlet noise 僅在 self-play 自我對弈 root 啟用
- Batched NN inference：leaf eval 進到一個待推論佇列，達到 batch size（128-256）或 timeout 才 flush 到 MLX
- Transposition：v1 暫不啟用（MCTS 下 transposition 有 subtle 問題，v2 再談）

### 3.3 Multi-Ponder（用戶要求的核心差異化功能）

**Critical file**：`engine/include/search/multi_ponder.hpp`

標準 UCI `ponder` 只算「對手最可能下的那一步」之後的延伸。用戶要的是**在對手思考時同步計算對手 top-5 回覆各自延伸的我方最佳步**。

演算法：
```
// 使用者選「single-side」+ multi-ponder=5 時
on_enter_opponent_turn():
    root = current_position
    # 快速 policy-only 評估（一次 NN forward，<30ms）
    top5_opp_moves = nn_policy(root).top_k(5)

    # Spawn 5 個搜索 task，權重預算
    weights = [0.40, 0.20, 0.15, 0.15, 0.10]
    for (move, w) in zip(top5_opp_moves, weights):
        child_pos = apply(root, move)
        tasks.spawn(SearchTask(root=child_pos, budget=total_budget * w))

    # 5 個 task 共享同一個 MLXBackend（batched inference 會跨 task 合併）

on_opponent_actually_played(move):
    if move in top5_opp_moves:
        tree = tasks[move].tree
        promote tree as new root        # 零延遲切換
        cancel other tasks
    else:
        abort all tasks
        start fresh search

on_my_turn_timeout_or_user_chose():
    report best move + top-3 MultiPV
```

**為什麼 MLX 統一記憶體讓這個特別有效**：5 棵樹 × 葉節點 inference 可以**跨樹 batch 合併**成一次 Metal kernel 呼叫，M3 10-core GPU 在 batch=256 時 TFLOP 利用率 >60%，比五個獨立推論省 3-4 倍時間。

### 3.4 通訊協議

**標準 UCI/USI 給相容性**：
- `setoption name MultiPV value 3` → Top-3 hint
- `go infinite` → 無限分析模式
- `go ponder` / `ponderhit` → 標準單線預想

**JSON-RPC 擴展給進階功能**：
- `engine/include/protocol/json_rpc_ext.hpp`
- method: `start_multi_ponder`, `opponent_played`, `set_side`, `get_eval_bar`
- event stream: `policy_preview`, `ponder_progress`, `best_move`, `ponder_hit`

文件：`shared/protocol_spec.md`（GUI 端與引擎端合約）

---

## 4. GUI（Pygame）設計

**Critical files**：
- `gui/src/gui/app.py` — Pygame 主迴圈
- `gui/src/gui/engine_client.py` — 起 C++ 引擎子進程、管線 UCI/USI + JSON-RPC
- `gui/src/gui/ponder_manager.py` — 5 棵 ponder tree 的 UI 狀態管理
- `gui/src/gui/analysis_panel.py` — eval bar、top-3 箭頭、PV 列表

**版面**：
```
┌───────┬─────────────────────┬──────────────────┐
│       │                     │ Eval  +0.35      │
│ Eval  │                     │ Depth 24 NPS 80K │
│ bar   │      Board          │ Nodes 2.1M       │
│ +0.35 │   (箭頭覆蓋 top-3)   │                  │
│       │                     │ Top-3 moves:     │
│       │                     │ 1. Nf3 +0.30     │
│       │                     │ 2. e4  +0.25     │
│       │                     │ 3. c4  +0.20     │
├───────┴─────────────────────┤                  │
│ 1. e4 e5 2. Nf3 Nc6 3. Bb5..│ Ponder trees:    │
├─────────────────────────────┤ ● e5  (40% time) │
│ [Analysis] [Single ▼] [Flip]│ ○ c5  (20%)      │
│ [New] [Save] [Load]         │ ○ e6  (15%) ...  │
└─────────────────────────────┴──────────────────┘
```

**功能對照（用戶需求 → 實作）**：
| 需求 | 實作 |
|---|---|
| 最佳前三步提示 | `setoption MultiPV 3` → analysis_panel 渲染半透明箭頭 |
| 目前場面優勢分數 | `info score cp <v>` 或 JSON-RPC `get_eval_bar` → 垂直 eval bar（正 = 白/先手優勢） |
| 可一直穩定計算下去 | `go infinite` / `go nodes ∞` → Pygame 維持 subprocess、持續接收 `info` 並刷新 |
| 單邊計算 | JSON-RPC `set_side: me=white` → 引擎只在 my turn 主動搜、對手 turn 做 multi-ponder |
| 對手回合預算 top-5 預想 | JSON-RPC `start_multi_ponder=5` → 5 棵樹並行跑，命中時秒出 |

---

## 5. Training Pipeline（Python + MLX）

**Critical files**：
- `training/src/training/models/transformer.py` — ChessShogiTransformer
- `training/src/training/data/encoding.py` — **與 C++ encoder bit-exact 對齊**（黃金測試）
- `training/src/training/data/chessbench_loader.py`
- `training/src/training/data/dlshogi_loader.py`
- `training/src/training/trainers/supervised.py`
- `training/src/training/trainers/selfplay.py`
- `training/src/training/export.py` — MLX weights → engine 讀的 safetensors
- `training/scripts/{pretrain,finetune_selfplay,evaluate}.py`

**MLX 優化關鍵**：
- `mx.compile(step, inputs=state, outputs=state)` 捕獲訓練步（~15-30% 吞吐提升）
- bf16 原生（M3 無 prefill penalty）
- `nn.average_gradients` 未來若多機時可用（現在只有一台 M3，先不開啟）

**Self-play 微調流程**：
1. Python orchestrator `scripts/finetune_selfplay.py`
2. 呼叫 C++ 引擎（透過 subprocess UCI）跑 self-play，引擎用**當前最新權重**
3. 每 5 K 局進 replay buffer，訓練一輪（~1 epoch on fresh data）
4. Arena vs 上一代權重 100 局，勝率 >55% 才取代
5. 迴圈

---

## 6. 專案結構

```
Chess_Game_mlx/
├── pyproject.toml            # uv workspace root（含 training + gui）
├── uv.lock
├── CMakeLists.txt            # C++ top-level（engine + tests + bench）
├── README.md
│
├── engine/                   # C++17
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── core/{game_rules,position,move,hash}.hpp
│   │   ├── chess/{chess_traits,chess_encoding,uci}.hpp
│   │   ├── shogi/{shogi_traits,shogi_encoding,usi}.hpp
│   │   ├── search/{mcts,node,multi_ponder,time_manager}.hpp
│   │   ├── nn/{backend,mlx_backend,batcher,input_encoder}.hpp
│   │   └── protocol/{uci_loop,usi_loop,multipv,json_rpc_ext}.hpp
│   ├── src/                  # 對應 .cpp
│   ├── tests/                # GoogleTest: perft, mcts, encoding
│   ├── bench/                # google-benchmark: perft, nn throughput
│   └── third_party/
│       ├── chess-library/    # git submodule, header-only MIT
│       ├── cshogi-core/      # git subtree, C++ subset MIT
│       └── mlx-c/            # git submodule, official MLX C API
│
├── training/                 # Python + MLX
│   ├── pyproject.toml        # uv 子包
│   ├── src/training/
│   │   ├── models/{transformer,smolgen,heads}.py
│   │   ├── data/{chessbench_loader,dlshogi_loader,encoding,selfplay_buffer}.py
│   │   ├── trainers/{supervised,selfplay,distillation}.py
│   │   ├── selfplay/{worker,arena}.py
│   │   ├── losses.py
│   │   └── export.py
│   ├── scripts/{download_chessbench,pretrain,finetune_selfplay,evaluate}.py
│   └── tests/
│
├── gui/                      # Python + Pygame
│   ├── pyproject.toml        # uv 子包
│   ├── src/gui/
│   │   ├── app.py
│   │   ├── board_renderer.py
│   │   ├── chess_board.py / shogi_board.py
│   │   ├── engine_client.py
│   │   ├── ponder_manager.py
│   │   ├── analysis_panel.py
│   │   ├── game_controller.py
│   │   └── themes.py
│   ├── assets/{pieces,fonts}/
│   └── tests/
│
├── shared/
│   ├── protocol_spec.md      # JSON-RPC 擴展協議
│   ├── encoding_spec.md      # NN input/output 編碼（chess + shogi）
│   └── golden_data/          # C++/Python 編碼對齊測試
│
└── benchmarks/
    ├── strength_test.py      # gauntlet matches
    └── perft_suite/          # 標準 perft 測試集
```

**工具鏈**：
- **uv** workspace（`pyproject.toml` 頂層 + `training/` `gui/` 子包）
- **CMake 3.25+**、Apple Clang 15+、C++17
- **pybind11**（讓 training 端需要時可直接呼叫 C++ self-play，避開 subprocess 開銷）
- **GoogleTest + Catch2**（擇一）給 C++ 測試
- **pytest** 給 Python 測試
- **ruff + mypy** Python lint
- **clang-format + clang-tidy** C++ lint

---

## 7. 實作階段（~5 個月單人全職，含訓練時間）

### Phase 0 — 基礎建設（Week 1-2）
- uv workspace、CMake top-level、MLX-C build（出 hello-world inference）
- Disservin/chess-library + cshogi C++ core 編譯整合、頭檔命名空間整理
- CI skeleton（macOS runner）
- **驗收**：`cmake --build && ctest` 綠燈；`uv run pytest` 綠燈

### Phase 1 — Chess 引擎 MVP（Week 3-6，與 Phase 2 並行）
- `chess/chess_traits.hpp` wrap chess-library
- `core/game_rules.hpp` 抽象
- 基本 MCTS（single-thread、random rollout 先跑通流程 → 再接 NN）
- UCI loop（position / go / bestmove / info multipv）
- **驗收**：perft(6) = 119 060 324 324 正確；`cutechess-cli` 可連線

### Phase 2 — Shogi 引擎 MVP（Week 4-8，並行）
- `shogi/shogi_traits.hpp` wrap cshogi core（含成駒、持駒、打入）
- USI loop
- **驗收**：perft 對照 YaneuraOu 官方測試集；`ShogiGUI` 可連線

### Phase 3 — NN 推論整合（Week 7-10）
- `nn/mlx_backend.hpp` 透過 mlx-c 載入 safetensors、batch inference
- Input encoder C++ ↔ Python 黃金測試（100 個位置 bit-exact）
- NN-guided PUCT-MCTS 接通
- 先載入 LC0 小網路驗證 policy 合法性（policy argmax 不能是非法走）
- **驗收**：C++ encoder 輸出與 Python encoder 輸出 MD5 相同；50 個隨機位置 policy top-1 match

### Phase 4 — Training Pipeline（Week 9-14，與 Phase 3 後半並行）
- `models/transformer.py` 完整模型（12 層、d=768）
- `data/chessbench_loader.py`（streaming from disk、bf16 batch）
- `data/dlshogi_loader.py`
- `trainers/supervised.py` + `mx.compile`
- `export.py`（MLX → safetensors 給 C++）
- **驗收**：小規模預訓練（100 M positions）→ Chess test loss 下降到 ~1.5；engine 可讀此權重對局不崩

### Phase 5 — Pygame GUI（Week 11-16，與 Phase 4 並行）
- `board_renderer.py`（chess 8×8 / shogi 9×9 + 手駒區）
- `engine_client.py`（subprocess stdin/stdout UCI + JSON 擴展）
- `analysis_panel.py`（eval bar、top-3 箭頭、PV、NPS）
- PGN (chess) / KIF (shogi) 讀寫
- **驗收**：對局 10 分鐘無 crash；top-3 顯示正確；eval 隨局面波動

### Phase 6 — Multi-Ponder（Week 15-18）
- `multi_ponder.hpp` 實作 5-tree 管理 + budget 分配
- `json_rpc_ext.hpp` + GUI `ponder_manager.py`
- 跨樹 batched NN inference
- **驗收**：對手回合 CPU + GPU 持續工作；當對手走 top-5 命中時，下一秒秒出 bestmove；未命中則正常重算

### Phase 7 — 訓練開跑（Week 17 onwards，與其他 phase 並行後持續）
- 完整 ChessBench 預訓練（500 M positions，~7-10 天）
- 完整 dlshogi 預訓練（~7-10 天）
- Self-play fine-tune（持續）
- 每月一次 gauntlet vs Stockfish 2500 / 2700 / 3000 等降階版

### Phase 8 — 進階功能（依時間餘裕）
- Transposition table（小心 MCTS 語意）
- NNUE 風格快速評估給 quiescence
- Opening book（Polyglot 格式 for chess、ShogiGUI 格式 for shogi）
- Endgame tablebase（Syzygy for chess）
- Lichess Bot / shogi club 24 對戰模組

---

## 8. 關鍵現成資源（借用、不重造輪子）

| 元件 | 來源 | 授權 | 用途 |
|---|---|---|---|
| Chess move-gen | [Disservin/chess-library](https://github.com/Disservin/chess-library) | MIT | header-only、Stockfish 驗證過、perft ready |
| Shogi move-gen | [TadaoYamaoka/cshogi](https://github.com/TadaoYamaoka/cshogi) C++ subset | MIT-style | bitboard、成駒、持駒、打入 |
| MLX | [ml-explore/mlx](https://github.com/ml-explore/mlx) | MIT | 訓練 + C++ 推論 |
| MLX-C | [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c) | MIT | C++ 呼叫 MLX |
| Training data (chess) | [google-deepmind/searchless_chess (ChessBench)](https://github.com/google-deepmind/searchless_chess) | Apache 2.0 | 10M 局、15B 標註 |
| Training data (shogi) | [dlshogi](https://github.com/TadaoYamaoka/DeepLearningShogi) teacher data | GPL（注意：訓練端接觸 OK，引擎不 link） | Shogi 監督式資料 |
| Test harness | [cutechess](https://github.com/cutechess/cutechess) / [ShogiGUI](http://shogigui.siganus.com/) | — | UCI/USI 相容測試 |
| 參考架構 | [LC0 BT4 paper](https://lczero.org/blog/2024/02/transformer-progress/) | — | Smolgen + transformer 設計 |
| 參考架構 | [DeepMind Grandmaster-Level Chess Without Search](https://arxiv.org/abs/2402.04494) | — | Supervised-only baseline |

**注意 GPL 傳染性**：dlshogi 權重 / 資料若要做 self-play 或蒸餾 fine-tune 到自己的引擎，**引擎產生的權重會繼承 GPL**。若你計劃開源為 MIT，要改為自家從零 self-play（犧牲速度）或只用 ChessBench（Apache）。

---

## 9. 驗證計劃（端到端）

### 9.1 正確性測試
- `engine/tests/perft_chess.cpp`：標準 6 個 perft 位置、depth 6，數字必須完全相符
- `engine/tests/perft_shogi.cpp`：YaneuraOu 公佈的 perft 對照表
- `shared/golden_data/`：100 個位置的 C++ / Python encoder 輸出 MD5 比對
- `engine/tests/mcts_tests.cpp`：PUCT 公式單元測試（已知分佈 → 預期選擇）

### 9.2 整合測試
- `cutechess-cli` 跑 100 局 self-engine vs self-engine（無 illegal move、無 crash）
- USI 相容：`ShogiGUI` + 我方引擎，10 局無問題
- GUI 冒煙：`uv run gui` 開局、走 20 步、開啟 analysis / ponder 模式，無 exception

### 9.3 強度測試（每月跑一次）
- `benchmarks/strength_test.py`：vs Stockfish 限階（2000、2500、2800 Elo，搭配 `skill level` 或 `uci_limitstrength`）
- 每對手 100 局、時控 1+0.6
- 估算 Elo（BayesElo 或 ordo）
- 目標里程碑：
  - Month 3：chess >2400 Elo，shogi 相當於 dlshogi-small
  - Month 5：chess >2700 Elo，shogi 超過 YaneuraOu pre-NNUE
  - Month 9+：chess >2900 Elo（搭配 MCTS 搜索）

### 9.4 功能驗收（手動 / UI 驗證）
對 chess 與 shogi 各跑一次：
1. 開啟 GUI，新局
2. 選 single-side = my white / 先手
3. 下 e4 / 2六歩，觀察：
   - eval bar 有數字、正值
   - top-3 箭頭正確顯示
   - 對手回合時 ponder panel 顯示 5 棵樹進度
   - 對手走對 top-5 其中一個時 → bestmove 在 <200 ms 出現
   - 未命中 → 正常思考時間
4. 切 analysis mode（`go infinite`），確認 NPS 穩定、可隨時停

---

## 10. 風險與應對

| 風險 | 應對 |
|---|---|
| MLX-C API 仍在演進、文件少 | Phase 0 就做 hello-world，若 API 有缺可 fallback libtorch MPS |
| C++ / Python encoder 不一致 | 黃金測試在 Phase 3 就鎖定、CI 每次 run |
| M3 Air 訓練比預期慢 | 降到 40 M 模型；用 ChessBench subset；必要時降級為從 LC0 蒸餾（違反「SOTA-beating」敘事但保強度） |
| Multi-ponder 跨樹 batching 實作複雜 | Phase 6 若落後，退一步先做標準 UCI ponder（單樹），multi-ponder 放到 v2 |
| cshogi C++ core 不易單獨編譯 | 備選：改用 [Apery](https://github.com/HiraokaTakuya/apery) 或 YaneuraOu move-gen（GPL 要注意） |
| 24 GB 不夠 batch size 256 | 啟用 gradient accumulation、降 batch；或換 40 M 模型 |

---

## 11. 第一週要做的事（讓用戶可直接開幹）

1. `uv init --package .`；新增 `training/` `gui/` 為 `members`
2. `git init`、加 `third_party` submodule：`chess-library`、`mlx-c`
3. `cmake -B build -DMLX_BUILD_METAL=ON`；跑 mlx-c sample
4. 寫 `engine/include/core/game_rules.hpp`（純 header，先放 TODO stub）
5. 寫 `engine/tests/perft_chess.cpp` 用 chess-library 的 perft API 跑通
6. 下載 ChessBench sample（~1 GB 抽樣）到 `data/chessbench/`
7. `training/src/training/models/transformer.py` 先實作可編譯的 40 M transformer、跑一次 forward + backward on fake batch，確認 MLX + bf16 正常

這七步大約 1-2 週、不需要 ML 強度考量，純工程暖身。

---

## 12. 與用戶後續決策點

以下在對應 phase 開始前再敲定即可，不影響整體計劃啟動：

- **Phase 3**：要不要用 pybind11 讓 training 端直接呼叫 C++ self-play（省 subprocess overhead），還是保持 subprocess 隔離簡化開發？
- **Phase 4**：要不要同時訓一個 ~10 M 的「小模型」專給 multi-ponder 的 policy preview（加快對手回合的 top-5 計算）？
- **Phase 7**：ChessBench 要取哪個 subset？完整 15 B 不需要，500 M 通常已夠強。
- **Phase 8**：要不要做 Lichess Bot 接口讓網路對戰當作最強的 Elo 估計？
