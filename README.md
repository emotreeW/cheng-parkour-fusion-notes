# Cheng parkour notes: how $z$ enters the actor

Isaac Gym, Unitree A1, Cheng extreme-parkour. Numbers from [`results/cheng_eval.jsonl`](results/cheng_eval.jsonl). Eval: 256 envs. Metric: `mxd_mean` = waypoints / 7.

Later work left this stack. These runs are the fusion table and the 0.18 gap that produced it.

## 1. Stack

Two stages: scandots teacher PPO, forward-depth student DAgger. Teacher frozen at `000-00` `model_0.pt`. Student camera default 192 envs, distill 5000 iter.

Student forward:

1. CNN maps $58\times 87$ depth to 32-d.
2. Concat with proprio, MLP, GRU $\to$ $z\in\mathbb{R}^{32}$ and yaw.
3. Actor concatenates `proprio(53) + z(32) + priv_explicit(9) + hist_latent(20)` into `[512, 256, 128] → 12`.
4. If `delta_yaw < 0.6`, the depth yaw head overwrites yaw in proprio.

Constraints: quadruped, sim-only, Isaac, A1; change who passes what to whom; no three-controller comparison; no 50 Hz $\to$ 200 Hz; no empty combo or PPO-loss swap as the whole paper.

## 2. Starting claim

Survey item 1 was “how visual $z$ enters the actor (concat / FiLM / hypernetwork, one paper)”:

- Cheng already concatenates $z$ into the MLP.
- REAL FiLMs visual features with proprio.
- Intended main method: FiLM proprio hidden states with $z$ (opposite direction to REAL).
- Variant: a small net generates the last linear $W$ from $z$.
- Control on this stack: REAL direction (proprio FiLMs CNN features), actor still concatenates.

No comparison row existed yet. Distill started with that claim already written as the method.

Out of scope for these runs: heatmap, MoE, CondConv, BEV. Teacher not retrained.

## 3. Teacher vs concat student

| run | role |
|---|---|
| `000-00-teacher` | scandots teacher, eval `model_0.pt` |
| `001-00-distill` | Cheng concat student. Load `model_4800.pt`, not this run’s `model_0.pt` (copy at distill start). |

`mxd_mean`:

| row | mixed | hurdle | step | gap | ramps |
|---|---:|---:|---:|---:|---:|
| teacher | 0.984 | 0.997 | 0.998 | 0.964 | 0.995 |
| concat (`concat` in jsonl) | 0.802 | 0.783 | 0.976 | 0.788 | 0.859 |
| concat, other eval (`distill`) | 0.803 | 0.785 | 0.972 | 0.772 | 0.874 |
| concat replicate | — | — | — | 0.777 | 0.842 |

Teacher mixed $\approx 0.98$, concat student $\approx 0.80$. Gap $\approx 0.18$, mostly mixed / hurdle / gap. That gap was treated as the place a different $z$ wiring might move.

## 4. Fusion modes

Teacher frozen at `000-00` checkpoint 0. Each mode 5000 iter. Terrains: mixed, hurdle, step, gap, ramps.

| mode | run | wiring | why it was trained |
|---|---|---|---|
| concat | `001-00` | $z$ concatenated into the actor | Cheng baseline, not retrained |
| film_prop | `002-00` | no concat $z$; first hidden $h\in\mathbb{R}^{512}$, $\tilde h=\gamma(z)\odot h+\beta(z)$ | intended main method: vision modulates proprio |
| hyper_last | `003-00` | no concat $z$; last $W\in\mathbb{R}^{12\times 128}$ generated from $z$ | variant. MorFiC groups FiLM and hypernets as multiplicative conditioning, so both rows sit in the table |
| film_vis | `004-00` | proprio FiLMs CNN features, actor still concatenates | REAL direction on this teacher / DAgger / eval, not REAL’s published numbers |
| film_cat | `005-00` | concat $z$ and FiLM the first hidden | after film_prop dropped ramps: test whether that drop came from removing $z$ from the input |

film_cat was added after the exclusive-or rule (concat XOR FiLM). ramps did not match “because $z$ was dropped,” so both were turned on together.

## 5. Fusion table

| row | mixed | hurdle | step | gap | ramps |
|---|---:|---:|---:|---:|---:|
| teacher | 0.984 | 0.997 | 0.998 | 0.964 | 0.995 |
| concat | 0.802 | 0.783 | 0.976 | 0.788 | 0.859 |
| concat replicate | — | — | — | 0.777 | 0.842 |
| film_prop | 0.797 | 0.772 | 0.967 | 0.783 | 0.758 |
| film_prop replicate | — | — | — | 0.791 | 0.768 |
| hyper_last | 0.655 | 0.707 | 0.953 | 0.709 | 0.647 |
| film_vis | 0.806 | 0.769 | 0.951 | 0.877 | 0.808 |
| film_vis replicate | — | — | — | 0.907 | 0.821 |
| film_cat | 0.776 | 0.856 | 0.979 | 0.869 | 0.796 |

Teacher-to-concat mixed gap $\approx 0.18$. Fusion deltas are about an order of magnitude smaller, except ramps and film_vis on gap.

film_prop matches concat on mixed / hurdle / step / gap; ramps falls to $\approx 0.76$. film_cat keeps $z$ in the input; ramps only returns to $0.80$, still below concat $0.84$–$0.86$. The ramps drop is not explained by dropping $z$ from the input.

hyper_last is below concat on all five terrains. film_vis mixed stays $\approx 0.80$; gap moves from $\approx 0.78$ to $\approx 0.88$–$0.91$.

Teacher weights inside the film_prop checkpoint match `000-00`. “Visual FiLM on proprio improves parkour” does not match the table.

## 6. What this table closed

- **$z$ into the actor as a paper method.** concat and film_prop mixed are tied; hyper_last is worse. Moving FiLM to another layer is the same wiring.
- **Hypernet as its own paper.** That is the hyper_last row.
- **Vision-FiLMs-proprio as its own paper.** That is film_prop; mixed is tied.
- **film_vis gap gain.** mixed did not move. REAL already FiLMs vision with proprio.

Survey items not trained on this table: contact MoE, CondConv, DELTA sampling on depth features, forward BEV with invalid cells, heatmap into the actor, xLSTM, fall-prediction head. They do not locate the 0.18. One distill is about 7.5 hours; they were not stacked.

## 7. Localizing the 0.18: `006-00-scandots_stu`

The large gap is teacher $0.98$ vs concat student $0.80$. The teacher reads scandots; the student reads $z$ from forward depth; both have heading. Split:

the 0.18 sits in depth vs scandots, or in the student joint net / DAgger; it may sit in heading; each piece may be smaller than 0.18.

`006-00-scandots_stu`: student actor takes the teacher’s `infer_scandots_latent` (32-d from 132-d `measured_heights`); yaw still comes from the depth head. Terrain input changes; heading does not. Teacher still frozen at `000-00`. Camera stays on for the yaw head.

`measured_heights` is privileged. Deployed A1 does not have those 132 numbers. Train and eval on this row use privileged height.

Planned mixed read (006 was not evaluated):

- $\approx 0.95$: gap mainly depth vs scandots. Next train: DELTA-style sampling on depth features vs Cheng global pool. Tie allowed.
- $\approx 0.80$: gap is not depth $z$ vs scandots $z$. Next: stop overwriting `obs[:, 6:8]` with predicted yaw. $\approx 0.95$ then means the yaw head; still $\approx 0.80$ means the joint net / DAgger. Either way, stop changing the vision module.
- in between: split the difference; no third fusion mode.

006 trained 5000 iter. No eval. Work left Cheng; this cell was not evaluated and did not start another vision module. Weights remain on the training box.

## 8. After Cheng

$z$-into-actor does not close the 0.18. 006, even with eval, would still be a privileged-height diagnostic. Later work moved to previous action $a_{t-1}$ on the Rudin stack, off Cheng, FiLM, hypernets, and 006.
