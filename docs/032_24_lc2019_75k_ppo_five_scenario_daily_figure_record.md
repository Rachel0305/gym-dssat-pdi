# 032_24 LC2019 75K PPO 五情景日过程图记录

## 结论先说

- 状态：完成。
- 本任务没有训练，只对 032_22 的 LC seed0 75K checkpoint 做冻结确定性回放，并保存 DSSAT snapshot。
- 代表年份：LC2019。选择理由：032_22 中 LC 75K/100K 在验证年均为 10/10 any-metric win；LC2019 同时具有产量和 PFP_N 相对四情景最高值的小幅优势，适合作为 LC 决策过程展示样本。

## PPO 候选摘要

| site | station | year | algorithm | seed | checkpoint | scenario | final_grain_kg_ha | final_biomass_kg_ha | rain_total_mm | irrigation_event_total_mm | nitrogen_event_total_kg_ha | max_water_stress_wspd | max_nitrogen_stress_nstd | common_reward_total | snapshot_path | etcp_mm | wp_et_kg_m3 | pfp_n_kg_kg | n_uptake_kg_ha | n_leaching_kg_ha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LC | LCA | 2019 | MaskablePPO | 0.0 | 75000.0 | rl_candidate | 9807.0 | 19505.0 | 206.8 | 150.0 | 240.0 | 0.0 | 0.012 | 5475.0 | benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/snapshots/LCA/2019/rl_candidate_seed0_ckpt75000 | 361.9 | 2.71 | 40.9 | 256.0 | 0.0 |

## Snapshot 完整性检查

| scenario | snapshot_path | exists | required_complete |
| --- | --- | --- | --- |
| null | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2019/null | True | True |
| recorded_farmer | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2019/recorded_farmer_template_02705 | True | True |
| dssat_auto | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2019/dssat_auto | True | True |
| official_extension_expert | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2019/official_extension_expert | True | True |
| rl_candidate | benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/snapshots/LCA/2019/rl_candidate_seed0_ckpt75000 | True | True |

## 输出文件

- daily CSV：`benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/tables/032_24_lc2019_75k_ppo_five_scenario_daily.csv`
- summary CSV：`benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/tables/032_24_lc2019_75k_ppo_five_scenario_summary.csv`
- evidence checks：`benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/tables/032_24_lc2019_75k_ppo_daily_evidence_checks.csv`
- snapshot checks：`benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/tables/032_24_snapshot_checks.csv`
- PPO snapshot：`benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/snapshots/LCA/2019/rl_candidate_seed0_ckpt75000`

## 图件

- `benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/figures/027_05_lc2019_maskableppo_five_scenario_daily.png`
- `benchmark_results/032_24_lc2019_75k_ppo_five_scenario_daily_figure/figures/027_05_lc2019_maskableppo_five_scenario_daily.svg`

## 备注

- recorded_farmer 使用 031_35 中的 `recorded_farmer_template_02705` snapshot。
- dssat_auto 使用 031_36 补齐后的 snapshot。
- 所有五情景均从 DSSAT snapshot 解析日过程，避免混用不完整 daily 表。
