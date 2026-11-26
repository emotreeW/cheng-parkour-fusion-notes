#!/bin/bash
# Table eval: 256 envs, camera on, five terrains.
# Concat uses 001-00 checkpoint 4800. Other modes load model_last.pt (checkpoint -1).
set -u
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
LOG="${LOG:-$ROOT/logs/cheng_fusion_eval.log}"
JSON="${EVAL_JSONL:-$ROOT/results/cheng_eval.jsonl}"
mkdir -p "$(dirname "$LOG")"
echo "===== FUSION EVAL START $(date -Is) =====" | tee -a "$LOG"

run_one() {
  local tag="$1" exptid="$2" ckpt="$3" mode="$4" terrain="$5" nenv="$6"
  echo "----- $tag $terrain nenv=$nenv ckpt=$ckpt mode=$mode $(date -Is) -----" | tee -a "$LOG"
  if "$PYTHON" -u "$ROOT/scripts/eval_one.py" \
      --eval_tag "$tag" --eval_exptid "$exptid" --eval_checkpoint "$ckpt" \
      --eval_terrain "$terrain" --eval_num_envs "$nenv" --eval_camera \
      --eval_fusion_mode "$mode" --eval_jsonl "$JSON" \
      >> "$LOG" 2>&1; then
    echo "OK $tag $terrain" | tee -a "$LOG"
    return 0
  fi
  echo "FAIL $tag $terrain rc=$?" | tee -a "$LOG"
  return 1
}

for t in mixed hurdle step gap ramps; do
  run_one concat 001-00 4800 concat "$t" 256 || run_one concat 001-00 4800 concat "$t" 192 || true
done

for spec in "film_prop:002-00-film_prop:film_prop" "hyper_last:003-00-hyper_last:hyper_last" "film_vis:004-00-film_vis:film_vis" "film_cat:005-00-film_cat:film_cat"; do
  tag="${spec%%:*}"
  rest="${spec#*:}"
  exptid="${rest%%:*}"
  mode="${rest##*:}"
  for t in mixed hurdle step gap ramps; do
    run_one "$tag" "$exptid" -1 "$mode" "$t" 256 || run_one "$tag" "$exptid" -1 "$mode" "$t" 192 || true
  done
done

echo "===== FUSION EVAL DONE $(date -Is) =====" | tee -a "$LOG"
"$PYTHON" "$ROOT/scripts/print_table.py" "$JSON" | tee -a "$LOG"
