# dlshogi Teacher Setup (Cross-Game Joint Distillation, shogi side)

This page is for users running the **Cross-Game Joint Distillation (XGD)**
research direction, where dlshogi serves as a teacher network for our shogi
labels (parallel to LC0/BT4 on the chess side).

---

## License — read this first

The dlshogi **engine** (`DeepLearningShogi`) is open source, but the **model
weights** (`model-dr2_exhi.zip`) are released under a *restrictive personal-use
license* by the author. The license **explicitly permits**:

- "Model learning from game records" — i.e. using it to generate labels for
  *your* model's training (this is exactly what we do).
- "Win-rate measurement" — using it as an opponent to estimate Elo.

The license **explicitly prohibits**:

- Redistribution of the model file (so we don't commit it to git).
- Modification or parameter reuse.
- Direct competitive deployment of the original model.
- Commercial use.

**Our XGD pipeline is the license-compliant use case.** We never bundle or
modify the model — we run it as a black-box teacher to label positions, then
train our own model on those labels. The trained student weights belong to
us.

The full license text is in Japanese at:
<https://tadaoyamaoka.hatenablog.com/entry/2021/08/17/000710>

If you don't agree to the license, **stop here** — fall back to the
floodgate-only pretraining path (`training/scripts/launch_shogi_v2.sh`),
which uses public game records and has no model-license issue.

---

## What you need

| Component | Source | Size | License |
|---|---|---|---|
| dlshogi USI engine binary | [dlshogi-wcsc32 release](https://github.com/TadaoYamaoka/DeepLearningShogi/releases/tag/wcsc32) | 12 MB | GPL v3 (engine) |
| dlshogi ONNX model | [dr2_exhi release](https://github.com/TadaoYamaoka/DeepLearningShogi/releases/tag/dr2_exhi) → `model-dr2_exhi.zip` | 51 MB zip → 54 MB ONNX | restrictive (above) |

The dr2_exhi zip is **password-protected**. The password is published on
the author's blog (the link above). We do not reproduce it here for license
hygiene — go look it up yourself.

---

## Install steps (macOS)

The dlshogi binaries on the GitHub releases are Windows-only. For macOS you
have three options:

### Option A: build from source (recommended)

```bash
cd /tmp
git clone --depth 1 https://github.com/TadaoYamaoka/DeepLearningShogi.git
cd DeepLearningShogi/usi
# dlshogi's build expects nvcc (CUDA) by default. For Mac use the ONNX-runtime
# variant — see the README's "OnnxRuntime version" section. Roughly:
make -f Makefile.onnx \
    ONNXRUNTIME_DIR=/opt/homebrew/Cellar/onnxruntime/1.25.0 \
    BLAS=OFF
# Result: ./bin/usi (the binary you'll point --engine at)
mkdir -p ~/dlshogi_engine
cp ./bin/usi ~/dlshogi_engine/dlshogi
```

### Option B: run the Windows binary under Wine

Not recommended (slow, fragile) — only useful if you already have Wine set up.

### Option C: use cshogi.usi.Engine as a thin USI wrapper

Skip the dlshogi binary entirely; just use the ONNX model with a Python USI
shim. We don't ship this — it's an exercise for the reader if Option A fails.

---

## Get the model

```bash
mkdir -p data/dlshogi_nets
cd data/dlshogi_nets

# Engine binaries (12MB, no model, GPL OK to commit but we ignore for cleanliness)
gh release download wcsc32 --repo TadaoYamaoka/DeepLearningShogi \
    --pattern "dlshogi-wcsc32.zip"

# Model file (51MB, restrictive license — read above first)
gh release download dr2_exhi --repo TadaoYamaoka/DeepLearningShogi \
    --pattern "model-dr2_exhi.zip"

# Extract the model (password-protected, find password on author's blog)
unzip -P "<password-from-blog>" model-dr2_exhi.zip
ls -lh model-dr2_exhi.onnx     # should be ~54 MB
```

The `.gitignore` already excludes `data/dlshogi_nets/*` so these files
won't accidentally land in commits.

---

## Verify it works

Quick smoke test that the engine + model load and respond to USI:

```bash
echo -e "usi\nsetoption name DNN_Model value $PWD/data/dlshogi_nets/model-dr2_exhi.onnx\nisready\nposition startpos\ngo nodes 64\nquit" | \
    ~/dlshogi_engine/dlshogi
```

You should see a `bestmove` line within a few seconds (after the ONNX session
compiles). If you see an error about `DNN_Model` not being a valid option,
your build of dlshogi may use a different option name (try `WeightsFile` or
`Network`); look at the upstream USI engine source for the option list.

---

## Generate distillation labels

Once the engine + model are ready, generate ~50k labelled shogi positions:

```bash
uv run python training/scripts/label_dlshogi.py \
    --engine ~/dlshogi_engine/dlshogi \
    --model  data/dlshogi_nets/model-dr2_exhi.onnx \
    --output data/dlshogi_labels_50k.npz \
    --num-positions 50000 \
    --multipv 32 \
    --nodes 64 \
    --softmax-temperature 1.5 \
    --seed 42
```

Expected throughput on M3 Air: ~5-8 positions/sec at the defaults; 50k
positions take ~2 hours. Use `--num-positions 5000` for a smoke test first
(15 minutes).

The output `.npz` has the same schema as our LC0 chess labels (`x`, `p`,
`v`, `m`, `p_top_idx`, `p_top_prob`), so the joint distillation trainer
consumes both files with a single code path.

---

## Then run the joint distillation

Once `data/dlshogi_labels_50k.npz` and `data/lc0_labels_soft_150k.npz` both
exist:

```bash
./training/scripts/launch_joint_distill.sh
```

See `training/scripts/launch_joint_distill.sh` for the full set of flags and
the hypothesis being tested.
