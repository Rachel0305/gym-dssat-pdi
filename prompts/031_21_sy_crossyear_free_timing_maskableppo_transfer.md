# 031_21 SY cross-year frozen transfer of free-timing MaskablePPO

## 背景与纠偏

031_20 只证明了当前自由时序 MaskablePPO 工程路径可以在多个站点分别训练运行；它不是“SY2014 模型跨年份迁移”，也不应作为跨站点泛化证据。

本任务回到当前真正的问题：

> 先固定 SYA2014 训练出的自由时序 MaskablePPO 模型，迁移到 SY 站其他已有四基线年份，比较其相对 null / recorded farmer / DSSAT auto / official extension expert 的产量、WP_ET、PFP_N 表现。

## 任务范围

- 不训练新模型。
- 不改 reward。
- 不改 action mask、预算上限、单次上限、7 天间隔、DAP 施氮截止等约束。
- 不迁移到其他站点。
- 只做 SY 同站跨年 frozen-policy 评估。

## 模型来源

使用已完成的 SYA2014 三个自由时序 MaskablePPO 模型：

- seed0: `benchmark_results/031_17_free_timing_discrete_maskableppo_ncost2x_scaled_reward_seed0/models/SYA/free_timing_discrete_maskableppo_ncost2x_scaled_reward_seed0.zip`
- seed1: `benchmark_results/031_18_free_timing_discrete_maskableppo_ncost2x_scaled_reward_seeds1_2/models/SYA/free_timing_discrete_maskableppo_ncost2x_scaled_reward_seed1.zip`
- seed2: `benchmark_results/031_18_free_timing_discrete_maskableppo_ncost2x_scaled_reward_seeds1_2/models/SYA/free_timing_discrete_maskableppo_ncost2x_scaled_reward_seed2.zip`

## 验证年份

本轮先使用已有四基线和 daily/snapshot 证据齐全的 SY 跨年年份：

- SYA2012
- SYA2015

四基线来自：

`benchmark_results/028_05_sy_crossyear_frozen_ppo_daily/028_05_all_cases_five_scenario_summary.csv`

只复用其中四个 baseline 情景行，不复用旧的 `rl_candidate` 行。

## 指标

对每个年份、每个 seed 的 frozen policy 运行一次 deterministic evaluation，并保存：

- final grain yield, kg/ha
- irrigation total, mm
- nitrogen total, kg/ha
- ETCP, mm
- WP_ET, kg/m3
- PFP_N, kg/kg；若 N=0，则记为 NaN，不参与 PFP_N 胜出判断
- max WSPD / max NSTD
- action sequence
- daily CSV
- DSSAT snapshot

WP_ET / PFP_N 必须从 DSSAT `Summary.OUT` 解析得到，不允许只用日值 CSV 手算替代。

## 判定

对每个 seed-year，将 RL candidate 与四基线最大值比较：

- `yield_strict_win`: RL yield > four-baseline max yield
- `wp_et_strict_win`: RL WP_ET > four-baseline max WP_ET
- `pfp_n_strict_win`: RL PFP_N > four-baseline positive-N max PFP_N
- `advisor_any_metric_strict_winner`: 三者任一为 True

本轮不设“必须 2/3 seed 成功”的扩展结论，只输出证据表。后续是否扩展到 SY 全部有天气年份，基于本轮记录决定。

## 停止线

- 若模型加载失败、环境 schema 不兼容、snapshot 指标无法解析，停止并记录失败，不改模型、不重训。
- 若只有产量和 PFP_N、没有 WP_ET，同样停止；不能用缺项指标冒充完整对照。
- 若候选表现不好，只记录结果，不现场调参。

