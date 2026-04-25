# Chess_Game_mlx — Usage Guide

完整的用法整理。最強配置：**BT4 ONNX (~3000-3400 Elo)** 經 ONNX Runtime + 我們的 MCTS。

---

## 1. 快速開局（90% 用戶想要的）

### 用 Pygame 跟 BT4 對弈
```bash
uv run python -m gui.app \
    --engine model \
    --model-path data/lc0_nets/BT4.onnx
```

副檔名自動偵測：
- `.onnx` → 走 LC0 路線（最強，3000+ Elo）
- `.safetensors` → 走我們自訓 ChessShogiTransformer（弱很多）
- `mock`（預設）→ 純隨機 mock engine（debug 用）

GUI 鍵位：`F` 翻盤 / `N` 新局 / `Esc` 退出。  
左上是評分條（白先手為正）、棋盤上會疊 top-3 半透明箭頭 + `#1 +0.35` 標籤、右側面板顯示 PV / depth / NPS、底部是 multi-ponder 樹進度。

### 純 CLI（無圖形、UCI mode）
```bash
./engine/bin/chess_engine --lc0-weights data/lc0_nets/BT4.onnx --threads 4
# 然後輸入：
uci
isready
position startpos
go movetime 3000     # 思考 3 秒
# 輸出：bestmove d2d4
```

---

## 2. CLI 旗標完整表

### `chess_engine`（UCI 模式）
| 旗標 | 作用 |
|---|---|
| `--lc0-weights PATH` | 載入 LC0 ONNX 權重（**最強**） |
| `--weights PATH` | 載入我們自訓的 .safetensors |
| `--threads N` | MCTS 工作 thread 數（建議 4） |
| `--multipv K` | 同時輸出 K 條最佳 PV（top-K hint） |
| 沒給 weights | StubBackend，~520 Elo，完全當 toy |

### `gui.app`（Pygame）
| 旗標 | 作用 |
|---|---|
| `--game chess|shogi` | 棋類（shogi UI 仍是 placeholder） |
| `--engine rule|model` | `rule` = mock 隨機；`model` = 真引擎 |
| `--model-path PATH` | 權重檔路徑（自動偵測 .onnx/.safetensors） |
| `--mode ai_ai|human_ai|human_human` | 對弈模式 |
| `--side white|black` | 你執哪一方（`human_ai` mode） |
| `--headless` | 不開視窗（CI 測試用，2 秒後退出） |

---

## 3. 主要工作流

### A. 用 GUI 跟 SOTA 引擎下棋
```bash
uv run python -m gui.app --engine model --model-path data/lc0_nets/BT4.onnx
```

### B. 自動 arena 測強度
```bash
uv run python benchmarks/arena.py \
    --a ./engine/bin/chess_engine \
    --a-args "--lc0-weights data/lc0_nets/BT4.onnx --threads 4" \
    --b stockfish \
    --b-setoption "setoption name UCI_LimitStrength value true" \
    --b-setoption "setoption name UCI_Elo value 2500" \
    --games 8 --movetime 3000 --opening-depth 6 --seed 42
```
輸出：`W-D-L (A vs B)` + 估計 Elo 差。

### C. JSON-RPC multi-ponder（手動互動）
```bash
./engine/bin/chess_engine --lc0-weights data/lc0_nets/BT4.onnx --threads 4
# 標準 UCI handshake
uci
isready
# 鎖定一邊（GUI 通常自動帶上）
{"jsonrpc":"2.0","method":"set_side","params":{"me":"white"},"id":1}
position startpos
go movetime 1000   # 我下白棋
# 對手回合 → 啟動 5-tree 預想
{"jsonrpc":"2.0","method":"start_multi_ponder","params":{"k":5},"id":2}
# 觀察 policy_preview + ponder_progress events 流出
{"jsonrpc":"2.0","method":"opponent_played","params":{"move":"e7e5"},"id":3}
# 命中 top-5 → ponder_hit + 立刻吐 bestmove
quit
```

### D. 訓練自家模型（已不需要、BT4 比較強）
**只在你想自己研究訓練 pipeline 時做**：

```bash
# 1. 用 LC0 標 100k positions（GPU 獨占下 ~30 分鐘）
uv run python training/scripts/label_lc0.py \
    --weights data/lc0_nets/t1_256_distilled.pb.gz \
    --output data/lc0_labels_100k.npz \
    --num-positions 100000 --nodes 1

# 2. 用 KL soft loss 訓練 40M 模型（~10 小時 on M3）
uv run python training/scripts/pretrain_lc0_soft.py \
    --data-path data/lc0_labels_100k.npz \
    --epochs 12 --batch-size 128 --lr 5e-5 \
    --d-model 512 --layers 12 \
    --out-dir checkpoints/my_model
```

---

## 4. 每個權重檔的強度對照

| 權重檔 | 大小 | 來源 | 估計 Elo | 使用 |
|---|---|---|---|---|
| `data/lc0_nets/BT4.onnx` | 707 MB | LC0 BT4 | **3000-3400** | `--lc0-weights ...` |
| `data/lc0_nets/t1_256_distilled.pb.gz` | 37 MB | LC0 t1_256 | ~2800 | 標訓練資料用，不直接跑 |
| `checkpoints/chess_40m_sf20k_e10/final.safetensors` | 161 MB | 自訓 40M / 20k SF labels | ~1100 | `--weights ...`（弱） |
| `checkpoints/chess_40m_lc0soft_150k/final.safetensors` | 161 MB | 自訓 40M / 150k LC0 KL | ~700 | 訓練炸了，不用 |
| StubBackend (no flag) | — | 隨機 policy | ~520 | 開發 debug |

**結論**：實戰用 `BT4.onnx`，唯一答案。

---

## 5. 性能與限制

| 項目 | BT4 ONNX | 我們的 40M MLX |
|---|---|---|
| nps（4 thread） | ~75 | ~1369 |
| 每步搜索品質 | 接近 LC0 | 接近隨機 |
| 等同實戰 Elo | 3000+ | 1100 |
| 推論 backend | ONNX Runtime CPU | MLX-C + Metal GPU |

NPS 只有 75 看起來很低，但 BT4 NN 太強，**少量 visits 就足以選出 GM 級走法**（2 sec 搜索就能挑 d2d4 走 Slav 主線）。Stockfish 跑 2M nps 但搜索演算法是 alpha-beta，跟 MCTS 不同維度。

### Metal 為什麼不開？
- ONNX Runtime CoreML provider 把 BT4 切成 51 段 partition、每次 forward 50 個 cross-provider handoff
- BT4 首次 CoreML 編譯 >10 分鐘
- 結果反而比 CPU 慢
- → 維持 CPU execution provider

---

## 6. 常見問題

**Q: GUI 開起來報 `Error: No such option: --lc0-weights`**  
A: GUI CLI 是 `--engine model --model-path ...`，不是 `--lc0-weights`（該旗標只給 `chess_engine` binary）。

**Q: `--engine ./engine/bin/chess_engine` 也錯**  
A: GUI 的 `--engine` 只接 `rule` / `model` 兩種選項。binary 路徑由 GUI 內部自動找 `engine/bin/chess_engine`。

**Q: 我的 GUI 下完一步、top-3 箭頭沒更新**  
A: 已修。確保拉的是最新版（`engine/include/protocol/uci_loop.hpp` 的 `handle_position` 會先 stop 舊搜索）。

**Q: 想要 shogi**  
A: `./engine/bin/shogi_engine` UCI/USI 完全可用。Pygame Shogi UI 還是 placeholder — 可用 third-party GUI（[ShogiGUI](http://shogigui.siganus.com/) / [ShogiHome](https://sunfish-shogi.github.io/shogihome/)）連我們的引擎。

**Q: BT4 推論很慢，能不能加速？**  
A: 三條路：
1. 用 t1_256_distilled (37MB, ~2800 Elo) 取代 BT4 — 快 10×、稍弱
2. 等 ONNX Runtime CoreML provider 改善 BT4 partitioning（看版本更新）
3. 加 GPU server fork ONNX Runtime + GPU EP（要編 source）

---

## 7. 從零跑通的最少步驟

```bash
# 一次性（如果還沒做）：
brew install lc0 stockfish onnxruntime
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCHESS_MLX_WITH_MLX=ON -DCHESS_MLX_BUILD_METAL=ON
cmake --build build -j

# BT4 onnx 已存在於 data/lc0_nets/BT4.onnx（707MB）
# 若沒有，從 lc0 weights 重新匯出：
# uv run python training/scripts/import_lc0.py --input data/lc0_nets/BT4.pb.gz --output data/lc0_nets/BT4.onnx

# 玩
uv run python -m gui.app --engine model --model-path data/lc0_nets/BT4.onnx
```

---

## 8. 系統測試

```bash
# C++ tests（133 tests, 約 4 sec）
ctest --test-dir build -j

# Python tests（81 tests, 約 11 sec）
uv run pytest -q

# 引擎 smoke test
echo -e 'uci\nposition startpos\ngo movetime 2000\nquit' | \
    ./engine/bin/chess_engine --lc0-weights data/lc0_nets/BT4.onnx --threads 4
# 應該看到 bestmove d2d4 / e2e4 / Nf3 等正規開局走法
```

---

## 9. 哪裡查更詳細的東西

| 你想做 | 看哪 |
|---|---|
| 整體計劃 / 架構 | `docs/original-plan.md` |
| LC0 整合內部細節 | `docs/lc0_import.md` |
| NN 編碼格式 | `shared/encoding_spec.md` |
| 自定義 JSON-RPC 協議 | `shared/protocol_spec.md` |
| 授權 | `LICENSE` (MIT for own code, GPL note for shogi + LC0) |
