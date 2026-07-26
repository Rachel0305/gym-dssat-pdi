# 036_04：036_01/036_03 结果的 checkpoint 选择与五站点汇总绘图

## 任务目的

在不重新训练、不修改 reward、不修改模型的前提下，整理 036_01 正式重跑与 036_03 replay 补齐 WP_ET 后的结果，形成一个可以汇报和复盘的阶段性结果包。

## 输入

- PPO replay 后完整验证表：
  - `benchmark_results/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018/tables/036_03_corrected_checkpoint_validation_with_wp_et.csv`
- checkpoint 站点聚合表：
  - `benchmark_results/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018/tables/036_03_by_station_checkpoint_with_wp_et.csv`
- 四情景基线表：
  - `benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_completed_template_aware_unified_baseline_summary.csv`
  - `benchmark_results/031_29_sy_auto_and_recorded_template_completion/evaluation/031_29_sy_baseline_summary.csv`

## 固定 checkpoint 选择规则

每个站点只选一个代表 checkpoint。选择规则在看图前固定如下：

1. `any_metric_win_count` 越大越优；
2. 若并列，`yield_win_count` 越大越优；
3. 若并列，`wp_et_win_count` 越大越优；
4. 若并列，`pfp_n_win_count` 越大越优；
5. 若仍并列，按 `mean_gap_yield + 1000 * mean_gap_wp_et + 10 * mean_gap_pfp_n` 越大越优；
6. 若仍并列，选择更早 checkpoint。

该规则只用于生成代表性汇总图，不改变所有 checkpoint 的原始记录。

## 输出

输出目录：

`benchmark_results/036_04_select_checkpoint_and_plot_03601_03603_summary/`

至少包含：

- `tables/036_04_selected_checkpoints.csv`
- `tables/036_04_selected_year_level_comparison.csv`
- `tables/036_04_selected_station_summary.csv`
- `figures/036_04_overall_water_n_saving_vs_expert.png`
- `figures/036_04_overall_water_n_saving_vs_expert.svg`
- 每站点一张：
  - `figures/036_04_<station>_selected_checkpoint_metric_gaps.png`
  - `figures/036_04_<station>_selected_checkpoint_metric_gaps.svg`
- 实验记录：
  - `docs/036_04_select_checkpoint_and_plot_03601_03603_summary_record.md`

## 边界

- 不训练。
- 不改 reward。
- 不改模型。
- 不删除或覆盖 036_01/036_02/036_03 原始输出。
- FQA2018 零产量异常保留，不静默删除。
- 若 baseline 有多个 recorded/template 版本，继续采用 036_02/036_03 的 available baseline max 口径，并在记录中说明。

