# Run table

Upstream: [chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour) `d2ffe27`.
Task: `a1`. Device: `cuda:0`. Distill always `--resume --resumeid 000-00 --checkpoint 0 --delay --use_camera --no_wandb --headless`.

Camera default is 192 environments. Eval uses 256. Smoke uses 64 environments and 2 iterations.

Logs go to `legged_gym/logs/parkour_new/<exptid>/`. Distill writes `model_{it}.pt` and `model_last.pt`. Loading `--checkpoint -1` prefers `model_last.pt`.

## Teacher

`000-00-teacher`, eval checkpoint `model_0.pt`. Trained with Cheng's base command (no camera). Not retrained for fusion.

```bash
python -u legged_gym/scripts/train.py --exptid 000-00-teacher --device cuda:0 --no_wandb --headless
```

## Distill, concat (Cheng baseline)

`001-00-distill`. Eval loads `model_4800.pt`. That run's `model_0.pt` is the copy at distill start.

```bash
python -u legged_gym/scripts/train.py \
  --exptid 001-00-distill \
  --resume --resumeid 000-00 --checkpoint 0 \
  --delay --use_camera --device cuda:0 --no_wandb --headless \
  --max_iterations 5000 --fusion_mode concat
```

## Fusion distill (5000 iterations)

```bash
# from repo root
bash scripts/train_fusion.sh film_prop 002-00-film_prop 5000
bash scripts/train_fusion.sh hyper_last 003-00-hyper_last 5000
bash scripts/train_fusion.sh film_vis 004-00-film_vis 5000
bash scripts/train_fusion.sh film_cat 005-00-film_cat 5000
```

Equivalent python:

```bash
cd legged_gym
python -u legged_gym/scripts/train.py \
  --exptid 002-00-film_prop \
  --resume --resumeid 000-00 --checkpoint 0 \
  --delay --use_camera --device cuda:0 --no_wandb --headless \
  --max_iterations 5000 --fusion_mode film_prop
```

## Scandots student (006)

Privileged `measured_heights`. Camera stays on for the yaw head.

```bash
bash scripts/train_scandots_stu.sh 006-00-scandots_stu 5000
```

## Eval

```bash
python scripts/eval_one.py \
  --eval_tag film_prop --eval_exptid 002-00-film_prop --eval_checkpoint -1 \
  --eval_terrain mixed --eval_num_envs 256 --eval_camera \
  --eval_fusion_mode film_prop

# concat student
python scripts/eval_one.py \
  --eval_tag concat --eval_exptid 001-00 --eval_checkpoint 4800 \
  --eval_terrain mixed --eval_num_envs 256 --eval_camera \
  --eval_fusion_mode concat

# teacher, no camera
python scripts/eval_one.py \
  --eval_tag teacher --eval_exptid 000-00 --eval_checkpoint 0 \
  --eval_terrain mixed --eval_num_envs 256
```

Five terrains: mixed, hurdle, step, gap, ramps. Batch script: `scripts/eval_fusion.sh`.

## Shape check (no Gym)

```bash
python scripts/check_fusion_dims.py
```
