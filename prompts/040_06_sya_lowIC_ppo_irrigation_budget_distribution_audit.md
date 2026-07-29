# 040_06 SYA lowIC PPO 灌溉预算分配审计

## 目的

040_04 显示 PPO 在 2014/2017 明确出现水分胁迫失败；040_05 进一步证明，在这两个年份中后期额外加一次灌溉可以显著提高产量并降低水分胁迫天数。

本任务不改训练、不重跑 DSSAT，只审计 PPO 最佳 checkpoint 75000 在验证年份 2014–2023 的灌溉预算分配：

> PPO 是否系统性把大部分灌溉预算用在 DAP1–30，导致后期水分胁迫时没有可用水量？

## 数据来源

- PPO daily CSV：来自 `040_00_checkpoint_validation_summary.csv` 中 checkpoint 75000 的 `daily_csv_path`。
- 年份诊断标签：来自 `040_04_ppo_vs_lowIC_baselines_by_year.csv`。

## 固定分段

灌溉量按以下 DAP 区间统计：

- `DAP1_30`
- `DAP31_60`
- `DAP61_90`
- `DAP91_plus`

同时统计：

- 各区间灌溉量和比例；
- 首次/末次灌溉 DAP；
- 水分胁迫首次出现 DAP；
- 水分胁迫天数；
- 胁迫首次出现前已使用的灌溉量；
- 胁迫首次出现后是否还有灌溉。

## 输出

- `benchmark_results/040_06_sya_lowIC_ppo_irrigation_budget_distribution_audit/tables/040_06_irrigation_distribution_by_year.csv`
- `benchmark_results/040_06_sya_lowIC_ppo_irrigation_budget_distribution_audit/tables/040_06_irrigation_distribution_by_diagnosis_class.csv`
- `docs/040_06_sya_lowIC_ppo_irrigation_budget_distribution_audit_record.md`

