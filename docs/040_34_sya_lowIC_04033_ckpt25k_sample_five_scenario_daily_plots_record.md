# 040_30 SYA lowIC 040_28 checkpoint75k 代表年份五情景日过程图记录

## 结论先说

- 成功绘图年份：3 / 3。
- 失败年份：0。
- 本任务没有训练、没有改 checkpoint、没有改输入；只把 040_28 seed0 checkpoint75000 的冻结评估结果画出来。
- PPO 的管理措施、WSPD/NSTD、产量/生物量来自 040_28 validation daily CSV。
- 四基线来自 039_02 lowIC 快照；recorded farmer 使用既有模板复用路径。
- 040_28 PPO daily CSV 不含 SoilWat SWTD，因此土壤水面板不伪造 PPO 线。
- 累计奖励面板使用统一 040_28 风格诊断奖励，避免旧 common reward 口径与本轮训练 reward 混用。

## PPO 管理事件

| requested_year | scenario | dap | irrigation_executed_mm | nitrogen_executed_kg_ha |
| --- | --- | --- | --- | --- |
| 2014 | rl_candidate | 1.0 | 30.0 | 120.0 |
| 2014 | rl_candidate | 8.0 | 30.0 | 120.0 |
| 2014 | rl_candidate | 40.0 | 45.0 | 0.0 |
| 2014 | rl_candidate | 47.0 | 45.0 | 0.0 |
| 2014 | rl_candidate | 61.0 | 45.0 | 0.0 |
| 2014 | rl_candidate | 68.0 | 45.0 | 0.0 |
| 2017 | rl_candidate | 1.0 | 30.0 | 120.0 |
| 2017 | rl_candidate | 8.0 | 30.0 | 120.0 |
| 2017 | rl_candidate | 31.0 | 45.0 | 0.0 |
| 2017 | rl_candidate | 40.0 | 45.0 | 0.0 |
| 2017 | rl_candidate | 61.0 | 45.0 | 0.0 |
| 2017 | rl_candidate | 68.0 | 45.0 | 0.0 |
| 2022 | rl_candidate | 1.0 | 30.0 | 120.0 |
| 2022 | rl_candidate | 8.0 | 30.0 | 120.0 |
| 2022 | rl_candidate | 40.0 | 45.0 | 0.0 |
| 2022 | rl_candidate | 47.0 | 45.0 | 0.0 |
| 2022 | rl_candidate | 61.0 | 45.0 | 0.0 |
| 2022 | rl_candidate | 68.0 | 45.0 | 0.0 |

## 五情景终值摘要

| site | station | station_code | year | algorithm | seed | checkpoint | scenario | final_grain_kg_ha | final_biomass_kg_ha | rain_total_mm | irrigation_event_total_mm | nitrogen_event_total_kg_ha | max_water_stress_wspd | max_nitrogen_stress_nstd | common_reward_total | snapshot_path | etcp_mm | wp_et_kg_m3 | pfp_n_kg_kg | n_uptake_kg_ha | n_leaching_kg_ha | source_status | WP_ET_kg_m3 | PFP_N_kg_kg | source_daily_csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SY | Shenyang | SYA | 2014 | MaskablePPO |  |  | null | 1573.0 | 3214.0 | 331.8 | 0.0 | 0.0 | 1.0 | 0.57 | -0.0 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2014/null | 295.1 | 0.53 |  | 29.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2014 | MaskablePPO |  |  | recorded_farmer | 3324.0 | 7411.0 | 331.8 | 0.0 | 586.0 | 0.969 | 0.014 | -2930.0 | benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/snapshots/lowIC/SYA/2014/recorded_farmer | 304.9 | 1.09 | 5.7 | 99.0 | 0.0 | existing_040_14_recorded_snapshot |  |  |  |
| SY | Shenyang | SYA | 2014 | MaskablePPO |  |  | dssat_auto | 2820.0 | 6114.0 | 320.5 | 230.1 | 0.0 | 0.0 | 0.504 | 1016.9 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2014/dssat_auto | 450.8 | 0.63 |  | 60.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2014 | MaskablePPO |  |  | official_extension_expert | 10840.0 | 18872.0 | 320.5 | 266.1 | 297.0 | 0.0 | 0.012 | 7515.9 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2014/official_extension_expert | 504.9 | 2.15 | 36.5 | 290.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2014 | MaskablePPO | 0.0 | 25000.0 | rl_candidate | 10148.3179 | 18088.7549 | 320.5 | 240.0 | 240.0 | 0.8214 | 0.1113 |  |  |  |  |  |  |  | authoritative_040_28_validation_daily_csv_no_swtd |  | 42.2847 | benchmark_results/040_33_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions/daily_outputs/SYA/SYA_2014_seed0_ckpt25000_daily.csv |
| SY | Shenyang | SYA | 2017 | MaskablePPO |  |  | null | 0.0 | 17.0 | 24.8 | 0.0 | 0.0 | 0.96 | 0.017 | -0.0 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2017/null | 24.4 | 0.0 |  | 0.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2017 | MaskablePPO |  |  | recorded_farmer | 0.0 | 17.0 | 24.8 | 0.0 | 242.0 | 0.961 | 0.017 | -1210.0 | benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/snapshots/lowIC/SYA/2017/recorded_farmer | 24.4 | 0.0 |  | 1.0 | 0.0 | existing_040_14_recorded_snapshot |  |  |  |
| SY | Shenyang | SYA | 2017 | MaskablePPO |  |  | dssat_auto | 2640.0 | 5445.0 | 267.2 | 273.4 | 0.0 | 0.0 | 0.513 | 2366.6 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2017/dssat_auto | 402.1 | 0.66 |  | 56.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2017 | MaskablePPO |  |  | official_extension_expert | 10896.0 | 18640.0 | 267.2 | 266.1 | 297.0 | 0.0 | 0.014 | 9144.9 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2017/official_extension_expert | 445.4 | 2.45 | 36.7 | 275.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2017 | MaskablePPO | 0.0 | 25000.0 | rl_candidate | 9060.1514 | 16799.5215 | 267.2 | 240.0 | 240.0 | 0.9381 | 0.0214 |  |  |  |  |  |  |  | authoritative_040_28_validation_daily_csv_no_swtd |  | 37.7506 | benchmark_results/040_33_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions/daily_outputs/SYA/SYA_2017_seed0_ckpt25000_daily.csv |
| SY | Shenyang | SYA | 2022 | MaskablePPO |  |  | null | 3774.0 | 6431.0 | 670.1 | 0.0 | 0.0 | 1.0 | 0.44 | -0.0 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2022/null | 372.4 | 1.01 |  | 65.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2022 | MaskablePPO |  |  | recorded_farmer | 10528.0 | 16701.0 | 670.1 | 0.0 | 586.0 | 1.0 | 0.012 | -2930.0 | benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/snapshots/lowIC/SYA/2022/recorded_farmer | 416.3 | 2.53 | 18.0 | 242.0 | 0.0 | existing_040_14_recorded_snapshot |  |  |  |
| SY | Shenyang | SYA | 2022 | MaskablePPO |  |  | dssat_auto | 4181.0 | 7354.0 | 670.1 | 62.4 | 0.0 | 0.0 | 0.43 | 344.6 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2022/dssat_auto | 417.1 | 1.0 |  | 79.0 | 0.0 |  |  |  |  |
| SY | Shenyang | SYA | 2022 | MaskablePPO |  |  | official_extension_expert | 10922.0 | 16182.0 | 670.1 | 266.1 | 297.0 | 0.0 | 0.012 | 5396.9 | benchmark_results/039_02_original_vs_lowIC_three_baseline_audit/snapshots/lowIC/SYA/2022/official_extension_expert | 461.8 | 2.36 | 36.8 | 256.0 | 26.0 |  |  |  |  |
| SY | Shenyang | SYA | 2022 | MaskablePPO | 0.0 | 25000.0 | rl_candidate | 10138.7805 | 15552.7002 | 672.7 | 240.0 | 240.0 | 0.0 | 0.0122 |  |  |  |  |  |  |  | authoritative_040_28_validation_daily_csv_no_swtd |  | 42.2449 | benchmark_results/040_33_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions/daily_outputs/SYA/SYA_2022_seed0_ckpt25000_daily.csv |

## 输出文件

- 日值表：`benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/tables/040_34_sya_2014_2017_2022_lowIC_04033_ckpt25k_five_scenario_daily.csv`
- 摘要表：`benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/tables/040_34_sya_2014_2017_2022_lowIC_04033_ckpt25k_five_scenario_summary.csv`
- 管理事件表：`benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/tables/040_34_sya_2014_2017_2022_lowIC_04033_ckpt25k_ppo_management_events.csv`
- 失败表：`benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/tables/040_30_failures.csv`

## 图件

- `benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/figures/040_34_lowIC_04033_ckpt25k_sy2014_five_scenario_daily.png`
- `benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/figures/040_34_lowIC_04033_ckpt25k_sy2017_five_scenario_daily.png`
- `benchmark_results/040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots/figures/040_34_lowIC_04033_ckpt25k_sy2022_five_scenario_daily.png`
