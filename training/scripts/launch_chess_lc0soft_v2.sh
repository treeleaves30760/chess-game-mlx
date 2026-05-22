#!/bin/zsh
# Chess 40M LC0-soft distillation — v2 retrain, reusing the 150k-labelled
# data from data/lc0_labels_soft_150k.npz (no Stage 1 LC0 labelling cost).
#
# Sidecar contract now respected by every step_*.json (see
# training/src/training/trainers/supervised.py::_arch_metadata).  Final
# weights land in checkpoints/chess_40m_lc0soft_v2/.

set -e
cd /Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx

OUT=checkpoints/chess_40m_lc0soft_v2
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 nohup .venv/bin/python -u training/scripts/pretrain_lc0_soft.py \
    --data-path data/lc0_labels_soft_150k.npz \
    --epochs 12 \
    --batch-size 128 \
    --lr 2e-4 \
    --warmup-steps 500 \
    --d-model 512 \
    --layers 12 \
    --out-dir "$OUT" \
    --seed 42 \
    --log-every 50 \
    --ckpt-every 500 > /tmp/chess_40m_lc0soft_v2_train.log 2>&1 &
disown
echo "chess v2 retraining launched, pid=$!"
echo "Log:        /tmp/chess_40m_lc0soft_v2_train.log"
echo "Checkpoints: $OUT/"
