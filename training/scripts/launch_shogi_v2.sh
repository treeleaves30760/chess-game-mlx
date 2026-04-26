#!/bin/zsh
# Restart 40M shogi training with the v2 schedule:
#   * Lower peak LR (1.5e-4) to stay safe with DeepNet init
#   * WSD schedule (warmup-stable-decay) — better fixed-budget convergence
#   * Longer warmup (2500 steps = 5%)
#   * Halved weight_decay (0.05) — less aggressive shrinkage
#   * Tighter grad_clip (0.5)
#   * Reduced w_value (0.3) — let policy lead early; value head needs more time
#   * Disabled w_moves_left (0.0) — hcpe has no per-position remaining-ply info
#
# Logs to /tmp/shogi_40m_v2_train.log; checkpoints in
# checkpoints/shogi_40m_floodgate_50k_v2/

set -e
cd /Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx

mkdir -p checkpoints/shogi_40m_floodgate_50k_v2

PYTHONUNBUFFERED=1 nohup .venv/bin/python -u training/scripts/pretrain.py \
  --game shogi \
  --steps 50000 \
  --batch-size 128 \
  --d-model 512 \
  --n-layers 12 \
  --n-heads 8 \
  --ffn-dim 2048 \
  --lr 1.5e-4 \
  --warmup-steps 2500 \
  --weight-decay 0.05 \
  --grad-clip 0.5 \
  --schedule wsd \
  --wsd-decay-frac 0.20 \
  --w-policy 1.0 \
  --w-value 0.3 \
  --w-moves-left 0.0 \
  --checkpoint-every 2000 \
  --checkpoint-dir checkpoints/shogi_40m_floodgate_50k_v2 \
  --data-dir data/dlshogi/floodgate_2025_R2700_train.hcpe \
  --log-every 25 \
  --seed 42 > /tmp/shogi_40m_v2_train.log 2>&1 &
disown
echo "v2 training launched, pid=$!"
echo "Log: /tmp/shogi_40m_v2_train.log"
echo "Checkpoints: checkpoints/shogi_40m_floodgate_50k_v2/"
