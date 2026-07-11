# 019_09 五站点已有证据状态刷新

## 目的

在不重新训练、不调用 DSSAT 的前提下，基于已经完成的 018_03、018_06、018_08、018_10、019_01、019_02 和 019_03–019_08 结果，刷新五站点当前决策表。

本任务不是重新进行优化空间审计，而是：

1. 核对五个站点的代表年份、DQN 结果、官方推广 expert、DSSAT auto 和跨 seed 状态；
2. 给每条结论保留可追溯的源 CSV；
3. 区分“产量占优”“投入量占优”和“WUE/NUE 已经证明”三种不同结论；
4. 接入 019_02 的离线 N-cost 重评分结论；
5. 接入 019_03–019_08 的淋洗变量与主奖励函数决策；
6. 只列出仍缺的最小实验，不启动新训练。

## 约束

- 不覆盖旧结果；
- 不修改正式奖励函数；
- 不修改任何站点初始条件；
- 不运行 Docker、DSSAT 或 DQN 训练；
- 中文记录；
- 发现证据冲突时保守降级，不凭记忆补数。

## 主要输入

- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_10_hla_yc_fq_seed_recheck/018_10_site_recheck_status.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_06_lc2010_seed_stability_audit/018_06_lc2010_seed_best_summary.csv`
- `DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/018_08_seed0_vs_seed1_comparison.csv`
- `DSSAT_auto_validation/reward_sensitivity_019_02/019_02_site_level_interpretation.csv`
- `docs/2026-07-10_019_08_keep_leaching_as_sensitivity_not_main_reward.md`

## 输出

- `DSSAT_auto_validation/five_site_strategy_design_019_01/019_09_current_five_site_decision_board.csv`
- `DSSAT_auto_validation/five_site_strategy_design_019_01/019_09_evidence_quality_and_gaps.csv`
- `DSSAT_auto_validation/five_site_strategy_design_019_01/019_09_source_inventory.csv`
- `DSSAT_auto_validation/five_site_strategy_design_019_01/019_09_data_quality_checks.csv`
- `docs/2026-07-10_019_09_five_site_evidence_status_refresh.md`
