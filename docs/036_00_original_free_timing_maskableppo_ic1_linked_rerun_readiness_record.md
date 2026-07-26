# 036_00 原自由时序 MaskablePPO：IC=1 + linked 动作接口正式重跑前审计记录

## 结论先说

- 配置检查通过：25/25。
- IC/输入源检查通过：97/97。
- linked 强制动作 smoke 通过：5/5。
- 总判定：通过，可以进入 036_01 正式重跑。

## 关键边界

- 036 主线恢复 032_22 原 stress-aware reward，不采用 035_04/035_06 的 reward 改动。
- 036 主线只承认两个底层修复：IC=1 输入链、DSSAT linked 管理模式。
- 本任务不训练 PPO，不评价最终农学优劣，只检查正式重跑前的地基是否干净。

## 配置检查

| check | actual | expected | pass |
| --- | --- | --- | --- |
| total_timesteps | 100000 | 100000 | True |
| checkpoint_steps | 25000,50000,75000,100000 | 25000,50000,75000,100000 | True |
| irrigation_levels | 0.0,15.0,30.0,45.0 | 0.0,15.0,30.0,45.0 | True |
| nitrogen_levels | 0.0,40.0,80.0,120.0 | 0.0,40.0,80.0,120.0 | True |
| season_irrigation_soft_limit | 160.0 | 160.0 | True |
| season_n_soft_limit | 250.0 | 250.0 | True |
| min_days_between_irrigation | 7 | 7 | True |
| min_days_between_fertilization | 7 | 7 | True |
| irrigation_allowed_dap_range | 1,120 | 1,120 | True |
| fertilization_allowed_dap_range | 1,90 | 1,90 | True |
| reward_type | harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001 | harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001 | True |
| yield_coef | 0.158 | 0.158 | True |
| water_cost | 1.1 | 1.1 | True |
| nitrogen_cost | 1.58 | 1.58 | True |
| water_stress_relief_coef | 10.0 | 10.0 | True |
| nitrogen_stress_relief_coef | 5.0 | 5.0 | True |
| reward_scale | 0.001 | 0.001 | True |
| ppo_learning_rate | 0.0003 | 0.0003 | True |
| ppo_gamma | 1.0 | 1.0 | True |
| ppo_gae_lambda | 1.0 | 1.0 | True |
| ppo_n_steps | 144 | 144 | True |
| ppo_batch_size | 144 | 144 | True |
| ppo_n_epochs | 5 | 5 | True |
| ppo_ent_coef | 0.01 | 0.01 | True |
| ppo_clip_range | 0.2 | 0.2 | True |

## IC/输入源按站点汇总

| station_code | rows | passed |
| --- | --- | --- |
| FQA | 19 | 19 |
| HLA | 20 | 20 |
| LCA | 19 | 19 |
| SYA | 19 | 19 |
| YCA | 20 | 20 |

## linked 强制动作 smoke

| station_code | year | null_summary_i | null_summary_n | forced_safe_i | forced_safe_n | forced_summary_i | forced_summary_n | forced_overview_is_linked | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2005 | 0.0 | 0.0 | 45.0 | 80.0 | 45.0 | 80.0 | True | True |
| HLA | 2004 | 0.0 | 0.0 | 45.0 | 80.0 | 45.0 | 80.0 | True | True |
| LCA | 2005 | 0.0 | 0.0 | 45.0 | 80.0 | 45.0 | 80.0 | True | True |
| SYA | 2005 | 0.0 | 0.0 | 45.0 | 80.0 | 45.0 | 80.0 | True | True |
| YCA | 2004 | 0.0 | 0.0 | 45.0 | 80.0 | 45.0 | 80.0 | True | True |

## 输出文件

- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_config_check.csv`
- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_ic_source_audit.csv`
- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_linked_forced_action_smoke.csv`

耗时：43.7 秒
