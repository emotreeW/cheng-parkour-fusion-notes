#!/bin/bash
# Usage: train_scandots_stu.sh <exptid> <max_iterations> [num_envs]
set -euo pipefail
EXPTID="${1:?exptid}"
MAXIT="${2:?max_iterations}"
NUMENV="${3:-}"
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

CMD=("$PYTHON" -u legged_gym/scripts/train.py
  --exptid "$EXPTID"
  --resume --resumeid 000-00 --checkpoint 0
  --delay --use_camera --device cuda:0 --no_wandb --headless
  --max_iterations "$MAXIT"
  --fusion_mode concat
  --student_visual scandots
)
if [[ -n "$NUMENV" ]]; then
  CMD+=(--num_envs "$NUMENV")
fi
echo "SCANDOTS_STU ${CMD[*]}"
"${CMD[@]}"
echo "SCANDOTS_STU done exptid=$EXPTID"
ls -lh "$ROOT/legged_gym/logs/parkour_new/${EXPTID}/" | tail -20
