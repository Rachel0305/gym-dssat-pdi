# 037_04：LC2019 early starter cap 50K 五情景日过程图记录

## 状态

- 完成。
- 本任务不训练，只对 037_03 的 LCA seed0 50K checkpoint 做冻结确定性评估并绘图。
- 四基线 snapshot 复用已有 LC2019 基线；RL candidate snapshot 本轮重新生成。

## RL candidate 摘要

| site | station | year | algorithm | seed | checkpoint | scenario | final_grain_kg_ha | final_biomass_kg_ha | rain_total_mm | irrigation_event_total_mm | nitrogen_event_total_kg_ha | max_water_stress_wspd | max_nitrogen_stress_nstd | common_reward_total | snapshot_path | etcp_mm | wp_et_kg_m3 | pfp_n_kg_kg | n_uptake_kg_ha | n_leaching_kg_ha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LC | LCA | 2019 | MaskablePPO | 0.0 | 50000.0 | rl_candidate | 9807.0 | 19505.0 | 206.8 | 30.0 | 40.0 | 0.0 | 0.012 | -230.0 | benchmark_results/037_04_lc2019_early_starter_cap_50k_five_scenario_daily/snapshots/LCA/2019/rl_candidate_seed0_ckpt50000_early_starter_cap | 338.2 | 2.9 | 245.2 | 256.0 | 0.0 |

## Snapshot 检查

| scenario | snapshot_path | exists | required_complete |
| --- | --- | --- | --- |
| null | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/LCA/2019/null | True | True |
| recorded_farmer | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/LCA/2019/recorded_farmer_template | True | True |
| dssat_auto | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/LCA/2019/dssat_auto | True | True |
| official_extension_expert | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/LCA/2019/official_extension_expert | True | True |
| rl_candidate | benchmark_results/037_04_lc2019_early_starter_cap_50k_five_scenario_daily/snapshots/LCA/2019/rl_candidate_seed0_ckpt50000_early_starter_cap | True | True |

## 输出文件

- daily CSV：`benchmark_results/037_04_lc2019_early_starter_cap_50k_five_scenario_daily/tables/037_04_lc2019_50k_early_cap_five_scenario_daily.csv`
- summary CSV：`benchmark_results/037_04_lc2019_early_starter_cap_50k_five_scenario_daily/tables/037_04_lc2019_50k_early_cap_five_scenario_summary.csv`
- evidence checks：`benchmark_results/037_04_lc2019_early_starter_cap_50k_five_scenario_daily/tables/037_04_lc2019_50k_early_cap_evidence_checks.csv`
- figure：`benchmark_results/037_04_lc2019_early_starter_cap_50k_five_scenario_daily/figures/027_05_lc2019_maskableppo_five_scenario_daily.png`
- figure：`benchmark_results/037_04_lc2019_early_starter_cap_50k_five_scenario_daily/figures/027_05_lc2019_maskableppo_five_scenario_daily.svg`

## 初步说明

- 037_03 50K 的 LC2019 策略为小剂量 starter：DAP1 灌溉 30mm、DAP2 施氮 40kg/ha。
- 本图用于检查该小剂量 starter 是否导致明显水氮胁迫、产量损失或过程异常。
