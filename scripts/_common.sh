# shellcheck shell=bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/rsl_rl:$ROOT/legged_gym:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
PYTHON="${PYTHON:-python}"
ulimit -l unlimited || true
cd "$ROOT/legged_gym"
