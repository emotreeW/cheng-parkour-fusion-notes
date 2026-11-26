# Cheng parkour notes: how $z$ enters the actor

Setup: Isaac Gym, Unitree A1, Cheng extreme-parkour. Eval uses 256 environments. The metric is `mxd_mean` (waypoints / 7). Raw rows are in [`results/cheng_eval.jsonl`](results/cheng_eval.jsonl). Commands are in [`configs/runs.md`](configs/runs.md).

We froze the scandots teacher at `000-00` `model_0.pt` and distilled a forward-depth student. The student–teacher mixed gap is about 0.18. Changing how the 32-d visual latent $z$ enters the actor did not close that gap. Later experiments left this stack.

## Code

This tree is a modified copy of [chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour) at `d2ffe27`. Paper: [arXiv:2309.14341](https://arxiv.org/abs/2309.14341). License: CC BY-NC 4.0 (`LICENSE`, `NOTICE`).

Isaac Gym is not included. Checkpoints are not included. The five files that differ from `d2ffe27` are listed in `NOTICE`. The full diff is [`patches/extreme-parkour-d2ffe27.diff`](patches/extreme-parkour-d2ffe27.diff).

```
rsl_rl/                 PPO, actor, depth backbone, DAgger runner
legged_gym/             A1 parkour env, train/play/evaluate
scripts/train_fusion.sh distill with --fusion_mode
scripts/eval_one.py     256-env eval, writes jsonl
scripts/check_fusion_dims.py  forward-shape check, no Gym
configs/runs.md         exact commands for each row
results/cheng_eval.jsonl
```

New flags:

- `--fusion_mode {concat,film_prop,hyper_last,film_vis,film_cat}`
- `--student_visual {depth,scandots}`
- Distill writes `model_last.pt`. `--checkpoint -1` loads that file if present.

## Install

Python 3.8, PyTorch 1.10+cu113, Isaac Gym Preview 4 (Ada 4090 works; Blackwell SM_120 does not). Then:

```bash
# download Isaac Gym from NVIDIA; default path $HOME/isaacgym/python
export ISAACGYM_PYTHON=$HOME/isaacgym/python
conda create -n parkour python=3.8
conda activate parkour
bash install.sh
```

`install.sh` pip-installs `isaacgym`, `rsl_rl`, and `legged_gym` editable. Optional: copy `scripts/env.example.sh` to `scripts/env.sh`.

Shape check without Gym:

```bash
python scripts/check_fusion_dims.py
```

## Train

Teacher (Cheng base policy, no camera). We used an existing `000-00` `model_0.pt` and did not retrain it.

```bash
cd legged_gym
python -u legged_gym/scripts/train.py --exptid 000-00-teacher --device cuda:0 --no_wandb --headless
```

Student distill, 5000 iterations, camera on, 192 environments by default:

```bash
bash scripts/train_fusion.sh concat 001-00-distill 5000
bash scripts/train_fusion.sh film_prop 002-00-film_prop 5000
bash scripts/train_fusion.sh hyper_last 003-00-hyper_last 5000
bash scripts/train_fusion.sh film_vis 004-00-film_vis 5000
bash scripts/train_fusion.sh film_cat 005-00-film_cat 5000
```

Smoke: `bash scripts/train_fusion.sh film_prop 002-smoke-film_prop 2 64`.

Privileged scandots student:

```bash
bash scripts/train_scandots_stu.sh 006-00-scandots_stu 5000
```

One Gym process per machine. Weights land in `legged_gym/logs/parkour_new/<exptid>/`.

## Eval

256 environments, 1500 steps, terrains mixed / hurdle / step / gap / ramps.

```bash
python scripts/eval_one.py \
  --eval_tag concat --eval_exptid 001-00 --eval_checkpoint 4800 \
  --eval_terrain mixed --eval_num_envs 256 --eval_camera \
  --eval_fusion_mode concat
```

`--eval_checkpoint -1` loads `model_last.pt`. Batch: `bash scripts/eval_fusion.sh`. Replicates on gap and ramps: `bash scripts/eval_rep.sh`. Print a table: `python scripts/print_table.py results/cheng_eval.jsonl`.

## 1. Stack

Training has two stages: scandots PPO for the teacher, DAgger on forward depth for the student. Distill runs 5000 iterations. With the camera on, the student uses 192 environments by default.

Student forward pass:

1. A CNN maps $58\times 87$ depth to 32 dimensions.
2. That vector is concatenated with proprioception, then an MLP and a GRU produce $z\in\mathbb{R}^{32}$ and a yaw estimate.
3. The actor concatenates `proprio(53) + z(32) + priv_explicit(9) + hist_latent(20)` and maps `[512, 256, 128]` to 12 joint targets.
4. If `delta_yaw < 0.6`, the depth yaw head overwrites yaw in proprioception.

Scope for this project: quadruped, simulation only, Isaac Gym, Unitree A1. We change who passes which tensor to whom. We do not compare three controllers, raise the control rate from 50 Hz to 200 Hz, or treat an empty module combo or a PPO-loss swap as the paper.

## 2. Starting claim

The survey’s first remaining item was how visual $z$ enters the actor (concat, FiLM, or a hypernetwork) as one paper.

Cheng already concatenates $z$ into the MLP. REAL applies FiLM to visual features using proprioception. We wrote the main method as the reverse: FiLM on proprioceptive hidden states, conditioned on $z$. A variant generates the last linear layer $W$ from $z$. The control on this stack is REAL’s direction (proprioception FiLMs CNN features) with concatenation kept.

No comparison row existed when distill started. Heatmaps, MoE, CondConv, and BEV were out of scope. The teacher was not retrained.

## 3. Teacher vs concat student

| run | role |
|---|---|
| `000-00-teacher` | scandots teacher; eval checkpoint `model_0.pt` |
| `001-00-distill` | Cheng concat student. Load `model_4800.pt`. This run’s `model_0.pt` is the copy at distill start. |

`mxd_mean`:

| row | mixed | hurdle | step | gap | ramps |
|---|---:|---:|---:|---:|---:|
| teacher | 0.984 | 0.997 | 0.998 | 0.964 | 0.995 |
| concat (`concat` in jsonl) | 0.802 | 0.783 | 0.976 | 0.788 | 0.859 |
| concat, other eval (`distill`) | 0.803 | 0.785 | 0.972 | 0.772 | 0.874 |
| concat replicate | n/a | n/a | n/a | 0.777 | 0.842 |

Teacher mixed is 0.984; concat student mixed is 0.802. The 0.18 gap is largest on mixed, hurdle, and gap. We treated that gap as the quantity a different $z$ wiring might change.

## 4. Fusion modes

The teacher stays at `000-00` checkpoint 0. Each mode trains for 5000 iterations. Terrains: mixed, hurdle, step, gap, ramps.

| mode | run | wiring | why we trained it |
|---|---|---|---|
| concat | `001-00` | $z$ concatenated into the actor | Cheng baseline; not retrained |
| film_prop | `002-00` | no concat $z$; first hidden $h\in\mathbb{R}^{512}$, $\tilde h=\gamma(z)\odot h+\beta(z)$ | intended main method: vision modulates proprioception |
| hyper_last | `003-00` | no concat $z$; last $W\in\mathbb{R}^{12\times 128}$ generated from $z$ | variant. MorFiC groups FiLM and hypernetworks as multiplicative conditioning, so both rows are in the table |
| film_vis | `004-00` | proprioception FiLMs CNN features; actor still concatenates | REAL’s direction under this teacher, DAgger, and eval protocol |
| film_cat | `005-00` | concat $z$ and FiLM the first hidden | after film_prop ramps fell: test whether the drop came from removing $z$ from the input |

We first forbade concat and FiLM together. film_prop ramps did not match the “$z$ was dropped” account, so `film_cat` turns both on.

## 5. Fusion table

| row | mixed | hurdle | step | gap | ramps |
|---|---:|---:|---:|---:|---:|
| teacher | 0.984 | 0.997 | 0.998 | 0.964 | 0.995 |
| concat | 0.802 | 0.783 | 0.976 | 0.788 | 0.859 |
| concat replicate | n/a | n/a | n/a | 0.777 | 0.842 |
| film_prop | 0.797 | 0.772 | 0.967 | 0.783 | 0.758 |
| film_prop replicate | n/a | n/a | n/a | 0.791 | 0.768 |
| hyper_last | 0.655 | 0.707 | 0.953 | 0.709 | 0.647 |
| film_vis | 0.806 | 0.769 | 0.951 | 0.877 | 0.808 |
| film_vis replicate | n/a | n/a | n/a | 0.907 | 0.821 |
| film_cat | 0.776 | 0.856 | 0.979 | 0.869 | 0.796 |

The teacher–concat mixed gap is 0.182. Most fusion changes vs concat are much smaller. The exceptions are ramps and film_vis on gap.

film_prop mixed is 0.797 vs concat 0.802. hurdle, step, and gap stay within 0.02 of concat. ramps is 0.758 vs concat 0.859 (replicate 0.842). film_cat keeps $z$ in the input; ramps is 0.796, still below concat. Removing $z$ from the input does not explain the ramps drop.

hyper_last is below concat on all five terrains (mixed 0.655 vs 0.802). film_vis mixed is 0.806. gap is 0.877 (replicate 0.907) vs concat 0.788 (replicate 0.777).

Teacher weights in the film_prop checkpoint match `000-00`. The claim that visual FiLM on proprioception raises parkour `mxd_mean` is inconsistent with this table.

## 6. Claims that stop here

concat and film_prop mixed differ by 0.005. hyper_last mixed is 0.147 below concat. Using $z$ to FiLM proprioception, or to generate the last layer, is not a separate method on this table. Moving FiLM to another layer is the same wiring.

film_vis raises gap by about 0.09 to 0.13 and leaves mixed at 0.806. REAL already FiLMs vision with proprioception.

We did not train contact MoE, CondConv, DELTA sampling on depth features, forward BEV with invalid cells, a heatmap into the actor, xLSTM, or a fall-prediction head on this table. Those designs do not locate the 0.18 gap. One distill takes about 7.5 hours; we did not stack them.

## 7. `006-00-scandots_stu`

The remaining question is where the 0.18 sits. The teacher reads scandots. The student reads $z$ from forward depth. Both have heading.

Hypothesis: the gap is in depth versus scandots, or in the student joint network and DAgger. It may also sit in heading. Each piece may be smaller than 0.18.

`006-00-scandots_stu` feeds the student actor the teacher’s `infer_scandots_latent` (32-d from 132-d `measured_heights`). Yaw still comes from the depth head. Terrain input changes; heading does not. The teacher stays at `000-00`. The camera stays on for the yaw head.

`measured_heights` is privileged. A deployed A1 does not observe those 132 values. Train and eval on this row use privileged height.

006 trained for 5000 iterations and was not evaluated. Planned mixed thresholds, if we had evaluated:

- About 0.95: the gap is mainly depth versus scandots. Next train would be DELTA-style sampling on depth features versus Cheng global pooling. A tie would stop that line.
- About 0.80: the gap is not depth $z$ versus scandots $z$. Next change would stop overwriting `obs[:, 6:8]` with predicted yaw. About 0.95 after that would point to the yaw head; still about 0.80 would point to the joint network or DAgger. Either outcome would stop further vision-module changes.
- Between 0.80 and 0.95: split the difference. Do not add a third fusion mode.

Work then left Cheng. Weights remain on the training machine.

## 8. After Cheng

$z$-into-actor does not close the 0.18 mixed gap (0.984 vs 0.802). 006, even with eval, would still use privileged height. Later work moved to previous action $a_{t-1}$ on the Rudin stack.
