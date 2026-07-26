# 034_02 PPO 外部动作落地链路根因审计记录

## 结论先说

- 判定：`management_mode_rr_blocks_external_action_and_ll_enables_it`。
- 本任务不训练、不修改原始输入，只做 FQA2014 的最小外部动作复核。

## 现有 034_01 快照模式解析

| file | exists | treatment | treatment_header | treatment_row | management_row | IC | MI | MF | PLANT | IRRIG | FERTI | source_scenario | overview_management_opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_results/034_01_ppo_action_dssat_effect_audit/snapshots/FQA/2014/null_external_noop/fileX.MZX | True | 1 | @N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM | 1 1 1 0 Sim2014                    1  1  0  1  1  1  1  0  0  0  0  0  1 | 1 MA              R     R     R     R     M | 1 | 1 | 1 | R | R | R | 034_01_null_external_noop | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M |
| benchmark_results/034_01_ppo_action_dssat_effect_audit/snapshots/FQA/2014/ppo_external_replay/fileX.MZX | True | 1 | @N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM | 1 1 1 0 Sim2014                    1  1  0  1  1  1  1  0  0  0  0  0  1 | 1 MA              R     R     R     R     M | 1 | 1 | 1 | R | R | R | 034_01_ppo_external_replay | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M |
| benchmark_results/034_01_ppo_action_dssat_effect_audit/snapshots/FQA/2014/ppo_static_mzx_same_schedule/fileX.MZX | True | 1 | @N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM | 1 1 1 0 Sim2014                    1  1  0  1  1  1  1  0  0  0  0  0  1 | 1 MA              R     R     R     R     M | 1 | 1 | 1 | R | R | R | 034_01_ppo_static_mzx_same_schedule | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M |
| DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2008/I60_N150_early/pdi_tmp_snapshot/fileX.MZX | True | 2 | @N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM | 2 1 1 0 Sim2008                    1  2  0  1  2  2  2  0  0  0  0  0  2 | 2 MA              R     L     L     R     M | 1 | 2 | 2 | R | L | L | known_linked_dqn_success_treatment2 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |

## 强制外部动作复核

| station_code | year | scenario | forced_first_action | mzx_irrig_mode | mzx_ferti_mode | overview_management_opt | safe_action_irrigation_sum_mm | safe_action_n_sum_kg_ha | summary_irrigation_mm | summary_n_kg_ha | grain_yield_kg_ha | max_swfac | max_nstres | snapshot_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2014 | external_rr | DAP1 I45/N80 | R | R | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :R  FERT :R  RESIDUE:R  HARVEST:M | 45.0 | 80.0 | 0.0 | 0.0 | 7953.631 | 0.0 | 0.3819 | benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/snapshots/FQA/2014/external_rr |
| FQA | 2014 | external_ll | DAP1 I45/N80 | L | L | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | 45.0 | 80.0 | 45.0 | 80.0 | 8295.5481 | 0.0 | 0.055 | benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/snapshots/FQA/2014/external_ll |

## 失败记录

无记录。

## 输出文件

- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_mode_parse.csv`
- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_forced_action_summary.csv`
- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_forced_action_daily.csv`

耗时：5.9 秒
