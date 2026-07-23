# 031_38 five-station PPO water/nitrogen saving summary

## Purpose

补齐 031_37 中缺失的 SY 站点，并生成五站点统一的 PPO 节水节氮汇总图和表。

## Boundary

- 不训练 PPO/DQN。
- 不运行 DSSAT。
- 只读取已有 PPO 迁移结果与已补齐四情景基线结果。
- 不覆盖 031_37 旧图，输出到新的 `031_38_five_station_ppo_water_n_saving_summary` 目录。

## Data sources

- FQA/HLA/LCA/YCA:
  - `benchmark_results/031_37_ppo_water_n_saving_summary/tables/031_37_all_seed_ppo_deltas_vs_official.csv`
  - `benchmark_results/031_37_ppo_water_n_saving_summary/tables/031_37_selected_station_year_ppo_deltas_vs_official.csv`
- SYA:
  - `benchmark_results/031_30_sy_ppo_vs_completed_baselines/evaluation/031_30_candidate_vs_baseline_envelope.csv`
  - Selection is recomputed with the same display rule used for 031_37.

## Selection rule

For display, select one PPO candidate per station-year by:

1. More strict metric wins among yield, WP_ET, and PFP_N against the four-baseline envelope.
2. Any strict metric winner first.
3. Higher `profit_simple`.
4. Higher final grain yield.
5. Lower irrigation.
6. Lower nitrogen.
7. Lower seed id.

## Metrics

- Four-baseline success is evaluated against the best value among null / DSSAT auto / official expert / recorded farmer template when available.
- Water and nitrogen saving are reported relative to `official_extension_expert`.
- This is intentional: null and DSSAT auto may use zero water or zero nitrogen, so they are not meaningful input-saving baselines.

## Outputs

- Per-station water/nitrogen saving plots.
- Overall five-station water/nitrogen saving plot.
- All-seed merged table.
- Selected station-year table.
- Station summary table with mean, standard deviation, min, max, and win counts.

