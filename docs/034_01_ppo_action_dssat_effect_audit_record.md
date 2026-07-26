# 034_01 PPO 动作是否真实进入 DSSAT 的三角审计记录

## 结论先说

| diagnosis | n |
| --- | --- |
| external_action_not_applied_to_dssat | 5 |

## 固定边界

- 不重新训练 PPO；只加载 033_04 已有 checkpoint。
- PPO 外部回放沿用 033_04 的 `StressAwareDiscreteWrapper` + action safety 路径。
- 静态回放使用同一 PPO safe-action 序列写入 `.MZX` 的 `@I/@F` 管理表，step 阶段只发送 no-op。
- PPO 代码层动作累计值与 DSSAT `Summary.OUT` 实际执行值分开报告。
- 本任务优先判定 PPO 外部 action 是否落地；静态 MZX 同序列回放只作为辅助参照。若多事件静态回放未完全等于 safe-action 总量，另立后续静态多事件格式修复，不影响外部 action 是否进入 DSSAT 的判定。

## 案例选择

| station_code | year | checkpoint_step | total_irrigation | total_n | action_sequence | model_path | model_exists |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2014 | 25000 | 45.0 | 80.0 | DAP1 I45/N80 | benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/models/FQA/FQA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | True |
| HLA | 2014 | 25000 | 150.0 | 240.0 | DAP1 I30/N40; DAP8 I30/N40; DAP15 I30/N40; DAP22 I30/N40; DAP29 I30/N40; DAP48 I0/N40 | benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/models/HLA/HLA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | True |
| LCA | 2014 | 25000 | 135.0 | 200.0 | DAP1 I45/N120; DAP8 I45/N40; DAP15 I45/N40 | benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/models/LCA/LCA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | True |
| SYA | 2014 | 25000 | 150.0 | 240.0 | DAP1 I45/N120; DAP8 I45/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | True |
| YCA | 2014 | 25000 | 150.0 | 240.0 | DAP1 I45/N40; DAP8 I0/N40; DAP9 I15/N0; DAP15 I0/N40; DAP16 I30/N0; DAP22 I0/N40; DAP23 I30/N0; DAP29 I0/N40; DAP30 I30/N0; DAP36 I0/N40 | benchmark_results/033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun/models/YCA/YCA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip | True |

## 三角审计 summary

| station_code | year | checkpoint_step | scenario | safe_action_irrigation_sum_mm | safe_action_n_sum_kg_ha | summary_irrigation_mm | summary_n_kg_ha | grain_yield_kg_ha | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 7953.631 | 0.0 | 0.3819 |
| FQA | 2014 | 25000 | ppo_external_replay | 45.0 | 80.0 | 0.0 | 0.0 | 7953.631 | 0.0 | 0.3819 |
| FQA | 2014 | 25000 | ppo_static_mzx_same_schedule | 90.0 | 160.0 | 45.0 | 80.0 | 8294.3604 | 0.0 | 0.0122 |
| HLA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 7333.8959 | 0.0 | 0.0148 |
| HLA | 2014 | 25000 | ppo_external_replay | 150.0 | 240.0 | 0.0 | 0.0 | 7333.8959 | 0.0 | 0.0148 |
| HLA | 2014 | 25000 | ppo_static_mzx_same_schedule | 180.0 | 280.0 | 30.0 | 40.0 | 7326.5906 | 0.0 | 0.0148 |
| LCA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 10524.2065 | 0.0 | 0.0122 |
| LCA | 2014 | 25000 | ppo_external_replay | 135.0 | 200.0 | 0.0 | 0.0 | 10524.2065 | 0.0 | 0.0122 |
| LCA | 2014 | 25000 | ppo_static_mzx_same_schedule | 180.0 | 320.0 | 45.0 | 120.0 | 10524.2065 | 0.0 | 0.0122 |
| SYA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 4361.4688 | 1.0 | 0.3786 |
| SYA | 2014 | 25000 | ppo_external_replay | 150.0 | 240.0 | 0.0 | 0.0 | 4361.4688 | 1.0 | 0.3786 |
| SYA | 2014 | 25000 | ppo_static_mzx_same_schedule | 195.0 | 360.0 | 45.0 | 120.0 | 6362.7332 | 0.9823 | 0.0631 |
| YCA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 8034.7528 | 0.7085 | 0.4531 |
| YCA | 2014 | 25000 | ppo_external_replay | 150.0 | 240.0 | 0.0 | 0.0 | 8034.7528 | 0.7085 | 0.4531 |
| YCA | 2014 | 25000 | ppo_static_mzx_same_schedule | 195.0 | 280.0 | 45.0 | 40.0 | 9145.1007 | 0.0 | 0.2971 |

## 对照判定

| station_code | year | checkpoint_step | ppo_safe_i | ppo_safe_n | external_summary_i | external_summary_n | static_summary_i | static_summary_n | null_yield | external_yield | static_yield | external_minus_null_yield | static_minus_null_yield | external_summary_matches_safe | static_summary_matches_safe | external_behaves_like_null | static_changes_dssat | external_summary_zero_despite_safe_action | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2014 | 25000 | 45.0 | 80.0 | 0.0 | 0.0 | 45.0 | 80.0 | 7953.631 | 7953.631 | 8294.3604 | 0.0 | 340.7294 | False | True | True | True | True | external_action_not_applied_to_dssat |
| HLA | 2014 | 25000 | 150.0 | 240.0 | 0.0 | 0.0 | 30.0 | 40.0 | 7333.8959 | 7333.8959 | 7326.5906 | 0.0 | -7.3053 | False | False | True | True | True | external_action_not_applied_to_dssat |
| LCA | 2014 | 25000 | 135.0 | 200.0 | 0.0 | 0.0 | 45.0 | 120.0 | 10524.2065 | 10524.2065 | 10524.2065 | 0.0 | 0.0 | False | False | True | True | True | external_action_not_applied_to_dssat |
| SYA | 2014 | 25000 | 150.0 | 240.0 | 0.0 | 0.0 | 45.0 | 120.0 | 4361.4688 | 4361.4688 | 6362.7332 | 0.0 | 2001.2643 | False | False | True | True | True | external_action_not_applied_to_dssat |
| YCA | 2014 | 25000 | 150.0 | 240.0 | 0.0 | 0.0 | 45.0 | 40.0 | 8034.7528 | 8034.7528 | 9145.1007 | 0.0 | 1110.3479 | False | False | True | True | True | external_action_not_applied_to_dssat |

## 失败记录

无记录。

## 输出文件

- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_case_selection.csv`
- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_summary.csv`
- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_daily.csv`
- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_comparison.csv`

耗时：52.5 秒
