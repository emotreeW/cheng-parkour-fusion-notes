#!/bin/bash
# Usage: train_fusion.sh <fusion_mode> <exptid> <max_iterations> [num_envs]
# num_envs omitted => camera default (192). Pass 64 for a smoke run.
set -euo pipefail
MODE="${1:?fusion_mode}"
EXPTID="${2:?exptid}"
MAXIT="${3:?max_iterations}"
NUMENV="${4:-}"
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

CMD=("$PYTHON" -u legged_gym/scripts/train.py
  --exptid "$EXPTID"
  --resume --resumeid 000-00 --checkpoint 0
  --delay --use_camera --device cuda:0 --no_wandb --headless
  --max_iterations "$MAXIT"
  --fusion_mode "$MODE"
)
if [[ -n "$NUMENV" ]]; then
  CMD+=(--num_envs "$NUMENV")
fi
echo "FUSION_TRAIN ${CMD[*]}"
"${CMD[@]}"
echo "FUSION_TRAIN done mode=$MODE exptid=$EXPTID"
ls -lh "$ROOT/legged_gym/logs/parkour_new/${EXPTID}/" | tail -20
