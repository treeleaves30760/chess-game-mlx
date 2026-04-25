#!/bin/zsh
# Overnight Route-1 pipeline: LC0 t1_256 soft-label distillation to 40M model.
#
# Stages:
#   1. Label 150,000 random chess positions with LC0 t1_256_distilled (~55 min @ 48 pos/s)
#   2. Train 40M ChessShogiTransformer for 12 epochs with KL-divergence loss (~10 h)
#   3. Arena vs Stockfish 1900 / 2200 / 2500 to measure actual Elo (~30 min)
#
# Wall-clock target: ~11 hours. Logs + checkpoints land under /tmp and checkpoints/.

set -e

cd /Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx

LOG_DIR=/tmp/route1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
DATA=data/lc0_labels_soft_150k.npz
CKPT=checkpoints/chess_40m_lc0soft_150k

echo "[$(date +%H:%M:%S)] Route 1 launch. Logs: $LOG_DIR" | tee "$LOG_DIR/run.log"
echo "[$(date +%H:%M:%S)] Data:   $DATA"                | tee -a "$LOG_DIR/run.log"
echo "[$(date +%H:%M:%S)] Weights out: $CKPT/"          | tee -a "$LOG_DIR/run.log"

# --- Stage 1: LC0 labelling ---------------------------------------------------
echo "\n[$(date +%H:%M:%S)] === STAGE 1: LC0 labelling (150k positions) ===" | tee -a "$LOG_DIR/run.log"
uv run python training/scripts/label_lc0.py \
    --weights data/lc0_nets/t1_256_distilled.pb.gz \
    --output  "$DATA" \
    --num-positions 150000 \
    --nodes 1 \
    --seed 42 \
    --log-every 5000 2>&1 | tee "$LOG_DIR/label.log"

echo "[$(date +%H:%M:%S)] Stage 1 complete." | tee -a "$LOG_DIR/run.log"

# --- Stage 2: KL-distillation training ---------------------------------------
echo "\n[$(date +%H:%M:%S)] === STAGE 2: 40M KL soft-distillation (12 epochs) ===" | tee -a "$LOG_DIR/run.log"
uv run python training/scripts/pretrain_lc0_soft.py \
    --data-path "$DATA" \
    --epochs 12 \
    --batch-size 128 \
    --lr 2e-4 \
    --warmup-steps 500 \
    --d-model 512 \
    --layers 12 \
    --out-dir "$CKPT" \
    --seed 42 \
    --log-every 50 \
    --ckpt-every 500 2>&1 | tee "$LOG_DIR/train.log"

echo "[$(date +%H:%M:%S)] Stage 2 complete." | tee -a "$LOG_DIR/run.log"

# --- Stage 3: Arena vs Stockfish ---------------------------------------------
echo "\n[$(date +%H:%M:%S)] === STAGE 3: Arena vs Stockfish ===" | tee -a "$LOG_DIR/run.log"

for ELO in 1600 1900 2200 2500; do
    echo "\n[$(date +%H:%M:%S)] --- vs SF Elo $ELO ---" | tee -a "$LOG_DIR/run.log"
    uv run python benchmarks/arena.py \
        --a ./engine/bin/chess_engine \
        --a-args "--weights $CKPT/final.safetensors --threads 4" \
        --b stockfish \
        --b-setoption "setoption name UCI_LimitStrength value true" \
        --b-setoption "setoption name UCI_Elo value $ELO" \
        --games 8 --movetime 1000 --opening-depth 6 --seed 42 \
        2>&1 | tee -a "$LOG_DIR/run.log"
done

echo "\n[$(date +%H:%M:%S)] === ALL STAGES DONE ===" | tee -a "$LOG_DIR/run.log"
echo "[$(date +%H:%M:%S)] Logs at: $LOG_DIR/"        | tee -a "$LOG_DIR/run.log"
echo "[$(date +%H:%M:%S)] Weights at: $CKPT/final.safetensors" | tee -a "$LOG_DIR/run.log"
