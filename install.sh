#!/bin/bash
# Isaac Gym is not in this tree. Download Preview 4 from NVIDIA first:
# https://developer.nvidia.com/isaac-gym
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAACGYM_PYTHON="${ISAACGYM_PYTHON:-$HOME/isaacgym/python}"

conda create -n parkour python=3.8
# conda activate parkour   # run this in your shell; conda activate is not safe in a script

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

if [[ ! -d "$ISAACGYM_PYTHON" ]]; then
  echo "Set ISAACGYM_PYTHON to isaacgym/python (got $ISAACGYM_PYTHON)" >&2
  exit 1
fi
(cd "$ISAACGYM_PYTHON" && pip install -e .)
(cd "$ROOT/rsl_rl" && pip install -e .)
(cd "$ROOT/legged_gym" && pip install -e .)
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
