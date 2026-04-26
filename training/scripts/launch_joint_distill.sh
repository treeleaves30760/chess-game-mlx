#!/bin/zsh
# Launch the **Cross-Game Joint Distillation (XGD)** training run.
#
# Hypothesis under test: a single 40M ChessShogiTransformer backbone trained
# jointly on chess (LC0/BT4 soft labels) and shogi (dlshogi soft labels)
# outperforms two independently-trained 40M models on each game alone.
#
# Prerequisites:
#   * data/lc0_labels_soft_150k.npz       (already produced by an earlier run)
#   * data/dlshogi_labels_50k.npz         (you must generate this — see
#                                          docs/dlshogi_setup.md)
#
# Schedule: WSD with 5% warmup, 20% linear decay tail. Peak LR 1.5e-4
# (matching v2). Batches alternate 50/50 chess/shogi by default — pass a
# different --shogi-weight to change the ratio.
#
# Logs to /tmp/xgd_joint_train.log; checkpoints in
# checkpoints/chess_shogi_xgd_joint_50k/

set -e
cd /Users/hsupohsiang/Self/Github_Local/Chess_Game_mlx

CHESS_DATA="data/lc0_labels_soft_150k.npz"
SHOGI_DATA="data/dlshogi_labels_50k.npz"

if [[ ! -f "$CHESS_DATA" ]]; then
    echo "ERROR: chess labels missing: $CHESS_DATA"
    echo "  Generate via training/scripts/label_lc0.py."
    exit 1
fi
if [[ ! -f "$SHOGI_DATA" ]]; then
    echo "ERROR: shogi labels missing: $SHOGI_DATA"
    echo "  See docs/dlshogi_setup.md to install dlshogi + run label_dlshogi.py."
    exit 1
fi

mkdir -p checkpoints/chess_shogi_xgd_joint_50k

PYTHONUNBUFFERED=1 nohup .venv/bin/python -u training/scripts/pretrain.py \
  --joint-distill \
  --chess-data "$CHESS_DATA" \
  --shogi-data "$SHOGI_DATA" \
  --interleave round_robin \
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
  --checkpoint-dir checkpoints/chess_shogi_xgd_joint_50k \
  --log-every 25 \
  --seed 42 > /tmp/xgd_joint_train.log 2>&1 &
disown

echo "XGD joint training launched, pid=$!"
echo "Log: /tmp/xgd_joint_train.log"
echo "Checkpoints: checkpoints/chess_shogi_xgd_joint_50k/"
echo ""
echo "Compare-against-baselines workflow:"
echo "  1. Wait for v2 to finish:    checkpoints/shogi_40m_floodgate_50k_v2/"
echo "  2. (Optional) train chess-only baseline if not done:"
echo "       pretrain.py --game chess --data-path \$CHESS_DATA ..."
echo "  3. Run XGD joint (this script)"
echo "  4. Arena all three on both games:"
echo "       shogi: training/scripts/shogi_arena.py --a nn:joint --b nn:shogi-only"
echo "              ...and --a nn:joint --b usi:dlshogi for ground truth"
echo "       chess: benchmarks/arena.py with chess_engine + each weight set"
