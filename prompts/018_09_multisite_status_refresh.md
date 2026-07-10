# 018_09 多站点状态刷新

## 目的

在 `018_08` 已补出 `SY2014 seed1` 证据后，刷新 `018_05` 的 multisite audit 口径，生成新的总表和 seed 状态表。

## 原则

- 不覆盖 `018_05` 旧文件
- 只新增一套刷新结果
- 以已有 CSV 为基础汇总，不新增训练，不额外消耗大算力

## 输入

- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_05_multisite_dqn_vs_extension_audit/018_05_site_level_audit.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_05_multisite_dqn_vs_extension_audit/018_05_pairwise_dqn_advantage.csv`
- `DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/018_08_seed0_vs_seed1_comparison.csv`

## 输出

- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_09_multisite_status_refresh/018_09_site_level_audit_refreshed.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_09_multisite_status_refresh/018_09_seed_status_summary.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_09_multisite_status_refresh/018_09_multisite_compact_status_table.csv`
- `docs/2026-07-09_018_09_multisite_status_refresh_record.md`
