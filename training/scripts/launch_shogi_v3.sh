#!/bin/zsh
# Shogi 40M re-training (v3) — kicks off after the supervised.py sidecar
# patch that writes feat_dim / num_moves / seq_len etc.  Output dir is
# shogi_40m_floodgate_50k_v3 to avoid touching the v2 partial checkpoints.
#
# Same hyperparams as v2; we're not chasing Elo here, just verifying the
# new sidecar pipeline end-to-end.  Logs at /tmp/shogi_40m_v3_train.log.

set -e
cd /Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx

mkdir -p checkpoints/shogi_40m_floodgate_50k_v3

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
  --checkpoint-dir checkpoints/shogi_40m_floodgate_50k_v3 \
  --data-dir data/dlshogi/floodgate_2025_R2700_train.hcpe \
  --log-every 25 \
  --seed 42 > /tmp/shogi_40m_v3_train.log 2>&1 &
disown
echo "shogi v3 retraining launched, pid=$!"
echo "Log:        /tmp/shogi_40m_v3_train.log"
echo "Checkpoints: checkpoints/shogi_40m_floodgate_50k_v3/"
