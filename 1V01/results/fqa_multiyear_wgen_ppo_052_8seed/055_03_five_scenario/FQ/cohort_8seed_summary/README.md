# FQ WGEN PPO 100K — 八个随机种子汇总

范围：PPO 训练种子 0–7；各模型在 2014–2023 固定历史天气上确定性评估；四个对照采用冻结的 051_03 结果。
每个种子的 PPO 10 年平均值为一个统计单位；表中 ±SD 表示 8 个训练种子之间的标准差。这里是描述性比较，不作显著性推断。
WP_ET 使用逐季 DSSAT Summary.OUT 的 ETCP 精确计算；PFP_N 沿用 Summary.OUT 口径；没有作物吸氮量时不报告 NUE。

## PPO 8 种子均值与四个冻结基线

| 指标 | PPO mean ± SD | PPO min–max | Null | Recorded | Auto + N | Expert | PPO 高于基线的种子数（Null / Recorded / Auto+N / Expert） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Yield (kg/ha) | 7234.25 ± 86.60 | 7118.40–7350.20 | 6953.40 | 7201.70 | 7178.20 | 7311.90 | 8/8 / 3/8 / 7/8 / 3/8 |
| WP_ET (kg/m³) | 2.14 ± 0.08 | 2.00–2.20 | 2.15 | 2.15 | 2.13 | 2.08 | 5/8 / 5/8 / 5/8 / 6/8 |
| PFP_N (kg grain/kg N) | 79.05 ± 55.25 | 33.87–197.74 | — | 55.58 | — | 33.16 | 0/8 / 5/8 / 0/8 / 8/8 |
| Irrigation (mm) | 110.62 ± 90.58 | 45.00–222.00 | 0.00 | 75.00 | 34.70 | 212.90 | 8/8 / 3/8 / 8/8 / 3/8 |
| Nitrogen (kg/ha) | 145.00 ± 82.64 | 40.00–240.00 | 0.00 | 144.00 | 0.00 | 241.70 | 8/8 / 3/8 / 8/8 / 0/8 |

## 解释边界

- 所有种子使用相同的 WGEN 训练 episode 调度，因此这里主要反映 PPO 初始化和优化随机性，不是 WGEN 天气池不确定性的独立重复抽样。
- 同一套 2014–2023 验证天气与四个冻结基线配对；seed 0–7 全部保留，不按结果选择‘最佳种子’。
- FQ 2018 原始 Tmin 异常按照冻结基线输入保留；相关图表和逐季快照可追溯。

## 文件

- `cohort_overall_mean_sd.csv`: 8 个种子的 10 年平均值及 4 个基线均值。
- `cohort_paired_10year_comparison.csv`: 每个 PPO seed 对四基线的 10 年配对差值。
- `cohort_by_year_mean_sd_and_baseline_deltas.csv`: 年度 PPO 均值、标准差及基线差。
- `ppo_seed_10year_means.csv`: 每个 seed 的 10 年均值。
- `all_seed_year_scenario_metrics.csv`: 全部 8×10×5 逐年情景指标。
- `cohort_8seed_mean_sd_vs_baselines.png` 和 `cohort_8seed_yearly_traces.png`: 整组图。
