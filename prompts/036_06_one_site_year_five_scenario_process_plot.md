# 036_06：生成一张当前PPO候选五情景过程对照图

## 任务目的

只生成一张用于备份和初步查看措施合理性的五情景过程图，不训练、不重跑DSSAT、不修改模型。

## 选择案例

- 站点年份：`LCA2021`
- PPO模型：036_04 选中的 LCA 代表 checkpoint，即 `checkpoint_step=100000`
- 选择理由：
  - LCA 在 036_04 中 10/10 年任一指标超过四情景最高值；
  - LCA2021 非零产量异常；
  - LCA2021 在代表 checkpoint 下产量和PFP_N均较高，适合作为“候选效果较好但措施需审计”的展示样例。

## 输入

- PPO daily：
  - `benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun/daily_outputs/LCA/LCA_2021_seed0_ckpt100000_daily.csv`
- 四情景 daily：
  - `benchmark_results/031_35_missing_four_baseline_completion_for_03134/evaluation/031_35_full_generated_baseline_daily.csv`
  - `benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_generated_dssat_auto_daily.csv`

## 统一累计奖励口径

不使用各情景源文件里的原始 `reward`，因为不同来源 reward 单位和尺度不完全一致。

本图统一使用：

```text
common_step_reward = ΔGRNWT - 1.1 × irrigation - 1.58 × nitrogen
```

然后按 DAP 累加为 `cumulative_common_reward`。

## 输出

- `benchmark_results/036_06_one_site_year_five_scenario_process_plot/figures/036_06_lca2021_five_scenario_process.png`
- `benchmark_results/036_06_one_site_year_five_scenario_process_plot/figures/036_06_lca2021_five_scenario_process.svg`
- `benchmark_results/036_06_one_site_year_five_scenario_process_plot/tables/036_06_lca2021_five_scenario_daily.csv`
- `benchmark_results/036_06_one_site_year_five_scenario_process_plot/tables/036_06_lca2021_five_scenario_summary.csv`
- `docs/036_06_one_site_year_five_scenario_process_plot_record.md`

## 边界

- 不训练；
- 不重跑DSSAT；
- 不改reward；
- 不改已有输出；
- 本图只作为阶段性备份和措施合理性查看样图，不能替代全站点过程审计。

