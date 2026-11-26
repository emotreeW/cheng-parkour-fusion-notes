# Cheng parkour：视觉 \(z\) 进 actor 对照表

Isaac Gym、Unitree A1、Cheng extreme-parkour 栈。教师冻 `000-00` 的 `model_0.pt`。学生 DAgger 5000 iter。评测 256 env。主指标 `mxd_mean` = 路标数 / 7。数字来自 `results/cheng_eval.jsonl`（评测日期 2026-11-24–26）。

这一组只说明：把 32 维视觉 \(z\) 从拼接改成 FiLM / 超网络，合不上教师到 concat 学生的 mixed 差（约 0.18）。**不够一篇 CCC。** 后面的选题已经离开 Cheng。

## 设定

- 教师：scandots PPO，路标 heading。
- concat 学生：深度 CNN-GRU → \(z\in\mathbb{R}^{32}\)，拼进关节 MLP；heading 由深度头预测，`delta_yaw < 0.6` 时覆盖本体 yaw。
- 教师不重训。学生视觉路默认深度图。

## `mxd_mean`

| 行 | mixed | hurdle | step | gap | ramps |
|---|---:|---:|---:|---:|---:|
| 教师 | 0.984 | 0.997 | 0.998 | 0.964 | 0.995 |
| concat（001-00） | 0.802 | 0.783 | 0.976 | 0.788 | 0.859 |
| concat 复评 | — | — | — | 0.777 | 0.842 |
| film_prop（视觉 FiLM 本体，不拼 \(z\)） | 0.797 | 0.772 | 0.967 | 0.783 | 0.758 |
| film_prop 复评 | — | — | — | 0.791 | 0.768 |
| hyper_last（\(z\) 生成最后一层 \(W\)） | 0.655 | 0.707 | 0.953 | 0.709 | 0.647 |
| film_vis（本体 FiLM 视觉特征，仍拼 \(z\)） | 0.806 | 0.769 | 0.951 | 0.877 | 0.808 |
| film_vis 复评 | — | — | — | 0.907 | 0.821 |
| film_cat（拼 \(z\) 且 FiLM 第一隐层） | 0.776 | 0.856 | 0.979 | 0.869 | 0.796 |

`distill` 是同一套 concat 学生的另一次评测（mixed 0.803 / hurdle 0.785 / step 0.972 / gap 0.772 / ramps 0.874），和 concat 行同一量级。

## 从这张表能读的

1. 教师到 concat 的 mixed 差约 0.18。fusion 各行之间，除 ramps 和 film_vis 的 gap 外，大约小一个数量级。
2. film_prop 的 mixed / hurdle / step / gap 和 concat 打平；ramps 掉到约 0.76。film_cat 把 \(z\) 留在输入里，ramps 只回到 0.80，仍低于 concat 的 0.84–0.86。
3. hyper_last 五块都低于 concat。
4. film_vis 的 mixed 仍约 0.80；gap 从约 0.78 到约 0.88–0.91，只在这一块地形上动得比较大。

## `006-00-scandots_stu`

学生 actor 吃教师 `infer_scandots_latent`（132 维 `measured_heights` 编成的 32 维），heading 仍走深度头。训练和评测都用特权高度。真机没有这 132 个数，这一行不能当部署方法，也不能当论文主方法。

原计划是用它定位 0.18 落在深度图相对 scandots，还是落在学生关节网络 / DAgger / yaw。选题已经离开 Cheng，**这一格不再评、不再据此开下一场视觉模块**。权重仍落在原训练目录，需要时再评。

## 不要写成

- film_prop / film_vis / hyper_last / film_cat 已经是方法。
- 学生部署时也能看 scandots。
- 这一张表够交 CCC。
