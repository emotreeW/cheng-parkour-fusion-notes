# Cheng parkour 实验记录：从「$z$ 进 actor」到离开 Cheng

Isaac Gym、Unitree A1、Cheng extreme-parkour。数字来自 [`results/cheng_eval.jsonl`](results/cheng_eval.jsonl)（评测 256 env，主指标 `mxd_mean` = 路标数 / 7）。这一组**不够一篇 CCC**。后面的选题已经离开 Cheng。

本文只写已经跑过的实验、当时为什么开、表出来之后读成什么。没有评测的格不写成已经成立的结果。

---

## 1. 栈和约束

Cheng 两阶段：教师 scandots PPO，学生前视深度 DAgger。教师冻 `000-00` 的 `model_0.pt`，不重训。学生开相机后默认 192 env，蒸馏 5000 iter。

学生前向：

1. CNN 把 $58\times 87$ 深度压成 32 维。
2. 和本体拼进 MLP，再 GRU，出 $z\in\mathbb{R}^{32}$ 和 yaw。
3. Actor 把 `proprio(53) + z(32) + priv_explicit(9) + hist_latent(20)` 拼进 `[512, 256, 128] → 12`。
4. `delta_yaw < 0.6` 时，深度头的 yaw 覆盖本体里的 yaw。

约束一直是：四足、纯仿真、Isaac、A1；改谁把什么传给谁；不比较三套控制器；不把 50 Hz 改成 200 Hz；空组合或换 PPO 损失不当整篇。目标会是 CCC 2027（初稿 2026-12-15）。

---

## 2. 调研交出去的题：把接法写成方法

调研二次确认留下 8 条能写项。第 1 条是「视觉 $z$ 怎么进 actor（concat / FiLM / 超网络，一篇）」：

- Cheng 已经把 $z$ 拼进 MLP。
- REAL 是本体去 FiLM 视觉特征。
- 本篇主方法写成：视觉 $z$ 去 FiLM 本体隐层（和 REAL 方向相反）。
- 变体：小网络用 $z$ 生成关节 MLP 最后一层 $W$。
- 杀增益行：同一栈上做 REAL 方向（本体 FiLM 视觉），actor 仍拼接。

交出去时对照行还没有。操控调研写过：现象表还没有就先定方法，失败过。这一轮等于把猜想当成了结论，再开 5000 iter。

当时明确不做：热图、MoE、CondConv、BEV；不重训教师。

---

## 3. 先有的两行：教师和 concat 学生

| run | 做什么 |
|---|---|
| `000-00-teacher` | scandots 教师。评测用 `model_0.pt`。 |
| `001-00-distill` | Cheng 默认 concat 学生。最后一份是 `model_4800.pt`，不要用这份 run 的 `model_0.pt`（蒸馏开始时的拷贝）。 |

`mxd_mean`：

| 行 | mixed | hurdle | step | gap | ramps |
|---|---:|---:|---:|---:|---:|
| 教师 | 0.984 | 0.997 | 0.998 | 0.964 | 0.995 |
| concat（jsonl 里 `concat`） | 0.802 | 0.783 | 0.976 | 0.788 | 0.859 |
| concat 另一次评测（`distill`） | 0.803 | 0.785 | 0.972 | 0.772 | 0.874 |
| concat 复评 | — | — | — | 0.777 | 0.842 |

教师 mixed 约 0.98，concat 学生约 0.80。差约 0.18，主要落在 mixed / hurdle / gap。当时把这个差当成「接法该打的地方」，去改 $z$ 怎么进 actor。

---

## 4. Fusion：四种接法，同一套蒸馏

代码加 `fusion_mode`。教师冻 `000-00` checkpoint 0。每档 5000 iter。评测五块地形：mixed / hurdle / step / gap / ramps。

| 档 | run | 接法 | 当时为什么开 |
|---|---|---|---|
| concat | `001-00` | $z$ 拼进 actor | Cheng 基线，不重训 |
| film_prop | `002-00` | 不拼 $z$；第一隐层 $h\in\mathbb{R}^{512}$ 上 $\tilde h=\gamma(z)\odot h+\beta(z)$ | **当时的主方法**：视觉调本体 |
| hyper_last | `003-00` | 不拼 $z$；最后一层 $W\in\mathbb{R}^{12\times 128}$ 由 $z$ 生成 | 变体。MorFiC 把 FiLM 和超网络算同一类乘法条件化，表里必须两行都在 |
| film_vis | `004-00` | 本体 FiLM CNN 特征，actor 仍拼接 | 杀增益行：同一教师、同一 DAgger、同一评测，不是去抄 REAL 论文里的数 |
| film_cat | `005-00` | 拼 $z$，并且 FiLM 第一隐层 | film_prop 的 ramps 掉了之后补的一刀：测「ramps 掉点是不是因为没把 $z$ 留在输入里」 |

film_cat 开场前，互斥约定是「禁止 concat 同时又 FiLM」。ramps 对不上「没拼 $z$」之后，才打破互斥，加这一行。

---

## 5. Fusion 表

| 行 | mixed | hurdle | step | gap | ramps |
|---|---:|---:|---:|---:|---:|
| 教师 | 0.984 | 0.997 | 0.998 | 0.964 | 0.995 |
| concat | 0.802 | 0.783 | 0.976 | 0.788 | 0.859 |
| concat 复评 | — | — | — | 0.777 | 0.842 |
| film_prop | 0.797 | 0.772 | 0.967 | 0.783 | 0.758 |
| film_prop 复评 | — | — | — | 0.791 | 0.768 |
| hyper_last | 0.655 | 0.707 | 0.953 | 0.709 | 0.647 |
| film_vis | 0.806 | 0.769 | 0.951 | 0.877 | 0.808 |
| film_vis 复评 | — | — | — | 0.907 | 0.821 |
| film_cat | 0.776 | 0.856 | 0.979 | 0.869 | 0.796 |

读这张表只用三句：

1. 教师到 concat 的 mixed 差约 0.18。这一档比 fusion 各行之间的差大约一个数量级。
2. film_prop 的 mixed / hurdle / step / gap 和 concat 打平；ramps 掉到约 0.76。film_cat 把 $z$ 留在输入里，ramps 只回到 0.80，仍低于 concat 的 0.84–0.86。ramps 掉点不能用「没拼 $z$」解释完。
3. hyper_last 五块都低于 concat。film_vis 的 mixed 仍约 0.80；gap 从约 0.78 到约 0.88–0.91，只在这一块地形上动得比较大。

代码不是坏的：film_prop 权重里教师部分和 `000-00` 对得上。主张「视觉 FiLM 本体能抬过障」对不上这张表。

---

## 6. 从表反过去：哪些题死了

- **视觉 $z$ 进 actor 当一篇方法**：停。concat 和 film_prop 的 mixed 打平，hyper_last 全面更差。再改 FiLM 插哪一层，是在已经判过的接法上加密。
- **超网络单独成篇**：就是 hyper_last。停。
- **反向 FiLM（视觉调制本体）单独成篇**：就是 film_prop。mixed 打平。停。
- **film_vis 的 gap 涨点**：留下边角，不够另开一篇。REAL 已经是本体 FiLM 视觉；mixed 没有跟着动。

调研里当时还留着、**没有用这张表开训**的：接触 MoE、CondConv、DELTA 深度采样、带无效格的前向 BEV、热图进 actor、xLSTM、摔倒预测头。它们回答不了「0.18 落在哪一段」。12 月 15 日前一场蒸馏大约 7.5 小时，不能把这些一次做完。

---

## 7. 改思考方式：先定位 0.18，再决定换不换接法

fusion 失败之后，错因写成：选题一开始就是结论。下一轮先写已经测到的现象，再写允许打平的猜想，哪一格的表还没有，就不把那一格的结果写进题目。

当时还在 Cheng 里，真正大的差是教师 0.98 对 concat 学生 0.80。教师读 scandots，学生读前视深度编出来的 $z$，两边都有 heading。于是把 0.18 拆成可以单独动的一段：

> 这 0.18 落在深度图相对 scandots，还是落在学生关节网络和 DAgger；允许落在 heading；允许各段都不够 0.18。

`006-00-scandots_stu` 就是这一刀：学生 actor 吃教师 `infer_scandots_latent`（132 维 `measured_heights` 编成的 32 维），yaw 仍走深度头。一次只换地形输入，不换 heading。教师仍冻 `000-00`。相机还开着，因为 yaw 头要深度。

这一行**不能当部署方法**。真机没有 `measured_heights`。训练和评测都用特权高度，只回答缺口在哪。

当时准备怎么读 mixed（评测没做，所以下面三条都不是结果）：

- 到约 0.95：0.18 主要在深度图相对 scandots。下一场只做 DELTA 深度特征采样，对照 Cheng 全局池化。采样允许打平。
- 仍约 0.80：0.18 不在「深度 $z$ 对 scandots $z$」这一段。下一场把 heading 改成不覆盖 `obs[:, 6:8]`。若这一下到约 0.95，缺口在 yaw 头；若仍约 0.80，缺口在学生关节网络 / DAgger。两种都停，不要再改视觉模块。
- 落在中间：把差拆开写，不要补第三种 fusion。

006 训到 5000。**没有评测。** 选题已经离开 Cheng，这一格不再评、不再据此开下一场视觉模块。权重仍在原训练目录，需要时再评。

---

## 8. 离开 Cheng

fusion 说明：$z$ 进 actor 合不上 0.18。006 即使评完，也只是特权高度上的定位，主方法那一行必须是部署时学生有的量。链路在 12 月 15 日前写不完就不交 Cheng 这篇。

后面换到 Rudin 栈上别的量（观测里的上一拍动作 $a_{t-1}$），和 Cheng 学生、FiLM、超网络、006 都无关。本仓库不再跟那条线。

---

## 不要写成

- film_prop / film_vis / hyper_last / film_cat 已经是方法。
- 学生部署时也能看 scandots。
- 这一张表够交 CCC。
- 006 已经证明 0.18 落在深度或落在 actor。没有评测。
