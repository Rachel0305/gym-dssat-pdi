# 034_05 FQA2014 linked 修复后自由时序 PPO/DQN 50K 对比记录

## 结论先说

- 接口闭环通过：1/1。
- 站点年份：FQA2014；seed=0；训练步数：50000。
- 本轮是 linked 管理修复后的单站点单年对比；仍不代表跨年或跨站点结论。
- MaskablePPO 50K 已完成并评估；DQN 50K 因长时间无结果，为节省算力已停止，本轮不报告 DQN 性能。

## 训练 summary

| algorithm | station_code | train_years | seed | total_timesteps | run_status | model_path | notes | task | linked_management_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | FQA | 2014 | 0 | 50000 | ok | benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/models/FQA/maskableppo_stress_aware_seed0.zip | PPO model completed before DQN runtime stop. | 034_05 | True |
| DQN | FQA | 2014 | 0 | 50000 | runtime_stopped |  | DQN 50K did not finish within the smoke waiting budget and was stopped to save compute; no result is reported. | 034_05 | True |

## RL 评估 summary

| algorithm | run_status | final_grnwt | total_irrigation | total_n | summary_irrigation_mm | summary_n_kg_ha | safe_summary_i_match | safe_summary_n_match | overview_is_linked | interface_pass | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | ok | 8080.127 | 150.0 | 240.0 | 150.0 | 240.0 | True | True | True | True | DAP1 I45/N40; DAP8 I45/N120; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 |

## 与四情景基线同口径比较

| model_type | scenario | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | etcp_mm | WP_ET_kg_m3 | PFP_N_kg_kg | simple_profit | delta_vs_best_baseline_grain_yield_kg_ha | delta_vs_best_baseline_WP_ET_kg_m3 | delta_vs_best_baseline_PFP_N_kg_kg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | null | 7953.631 | 0.0 | 0.0 | 351.6 | 2.26 |  | 7953.631 | -364.7638 | -0.06 |  |
| baseline | recorded_farmer_template | 8273.1494 | 75.0 | 144.0 | 364.1 | 2.27 | 57.5 | 7963.1294 | -45.2454 | -0.05 | -43.9 |
| baseline | official_extension_expert | 8318.3948 | 23.0 | 82.0 | 359.0 | 2.32 | 101.4 | 8163.5348 | 0.0 | 0.0 | 0.0 |
| baseline | dssat_auto | 7953.631 | 0.0 | 0.0 | 351.6 | 2.26 |  | 7953.631 | -364.7638 | -0.06 |  |
| RL | linked_free_timing_maskableppo_50k | 8080.127 | 150.0 | 240.0 | 363.8 | 2.22 | 33.7 | 7535.927 | -238.2678 | -0.1 | -67.7 |

## 失败/停止记录

| phase | algorithm | status | traceback |
| --- | --- | --- | --- |
| train | DQN | runtime_stopped | No exception. Process was manually stopped after PPO completed and DQN 50K produced no model within the smoke waiting budget. |

## 输出文件

- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_training_summary.csv`
- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_eval_summary.csv`
- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_scenario_comparison.csv`

耗时：6.4 秒。
