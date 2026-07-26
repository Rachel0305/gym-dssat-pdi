# 034_03 动态 RL linked management 修复 smoke 记录

## 结论先说

- 通过数：5/5。
- 本任务不训练，只验证修复后动态 PPO 外部动作是否真正进入 DSSAT。

## Summary

| station_code | year | checkpoint_step | scenario | safe_action_irrigation_sum_mm | safe_action_n_sum_kg_ha | summary_irrigation_mm | summary_n_kg_ha | grain_yield_kg_ha | overview_management_opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 7953.631 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| FQA | 2014 | 25000 | ppo_external_replay | 45.0 | 80.0 | 45.0 | 80.0 | 8295.5481 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| HLA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 7333.8959 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| HLA | 2014 | 25000 | ppo_external_replay | 150.0 | 200.0 | 150.0 | 200.0 | 7200.0842 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| LCA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 10524.2065 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| LCA | 2014 | 25000 | ppo_external_replay | 135.0 | 200.0 | 135.0 | 200.0 | 10497.5879 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| SYA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 4361.4688 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| SYA | 2014 | 25000 | ppo_external_replay | 150.0 | 240.0 | 150.0 | 240.0 | 10488.8123 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| YCA | 2014 | 25000 | null_external_noop | 0.0 | 0.0 | 0.0 | 0.0 | 8034.7528 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |
| YCA | 2014 | 25000 | ppo_external_replay | 150.0 | 240.0 | 150.0 | 240.0 | 9391.2964 | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M | MANAGEMENT OPT : PLANTING:R  IRRIG   :L  FERT :L  RESIDUE:R  HARVEST:M |

## 判定表

| station_code | year | checkpoint_step | ppo_safe_i | ppo_safe_n | ppo_summary_i | ppo_summary_n | null_summary_i | null_summary_n | null_zero | ppo_summary_matches_safe | overview_is_linked | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2014 | 25000 | 45.0 | 80.0 | 45.0 | 80.0 | 0.0 | 0.0 | True | True | True | True |
| HLA | 2014 | 25000 | 150.0 | 200.0 | 150.0 | 200.0 | 0.0 | 0.0 | True | True | True | True |
| LCA | 2014 | 25000 | 135.0 | 200.0 | 135.0 | 200.0 | 0.0 | 0.0 | True | True | True | True |
| SYA | 2014 | 25000 | 150.0 | 240.0 | 150.0 | 240.0 | 0.0 | 0.0 | True | True | True | True |
| YCA | 2014 | 25000 | 150.0 | 240.0 | 150.0 | 240.0 | 0.0 | 0.0 | True | True | True | True |

## 失败记录

无记录。

## 输出文件

- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_summary.csv`
- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_comparison.csv`

耗时：37.4 秒
