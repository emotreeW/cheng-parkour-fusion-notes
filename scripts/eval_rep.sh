#!/bin/bash
# Replicates on ramps and gap for concat, film_prop, film_vis.
set -u
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
LOG="${LOG:-$ROOT/logs/cheng_fusion_eval_rep.log}"
JSON="${EVAL_JSONL:-$ROOT/results/cheng_eval.jsonl}"
mkdir -p "$(dirname "$LOG")"
echo "===== FUSION EVAL REP START $(date -Is) =====" | tee "$LOG"

run_one() {
  local tag="$1" exptid="$2" ckpt="$3" mode="$4" terrain="$5"
  echo "----- $tag $terrain ckpt=$ckpt mode=$mode $(date -Is) -----" | tee -a "$LOG"
  if "$PYTHON" -u "$ROOT/scripts/eval_one.py" \
      --eval_tag "$tag" --eval_exptid "$exptid" --eval_checkpoint "$ckpt" \
      --eval_terrain "$terrain" --eval_num_envs 256 --eval_camera \
      --eval_fusion_mode "$mode" --eval_jsonl "$JSON" >> "$LOG" 2>&1; then
    echo "OK $tag $terrain" | tee -a "$LOG"
  else
    echo "FAIL $tag $terrain rc=$?" | tee -a "$LOG"
  fi
}

run_one concat_r2 001-00 4800 concat ramps
run_one film_prop_r2 002-00-film_prop -1 film_prop ramps
run_one concat_r2 001-00 4800 concat gap
run_one film_vis_r2 004-00-film_vis -1 film_vis gap
run_one film_prop_r2 002-00-film_prop -1 film_prop gap
run_one film_vis_r2 004-00-film_vis -1 film_vis ramps
echo "===== FUSION EVAL REP DONE $(date -Is) =====" | tee -a "$LOG"
