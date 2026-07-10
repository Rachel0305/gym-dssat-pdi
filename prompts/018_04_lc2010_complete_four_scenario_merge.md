# 018_04 LC2010 完整四情景补齐与官方推广 expert 合并

## 背景

018_03 已扩展官方农技推广 expert baseline 到五站点代表年份，但 LC2010 在合并表中不完整：null / recorded / DSSAT auto 只使用了旧脚本常量，缺少生物量、灌溉、施氮和胁迫等完整字段。

经检查，`DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/017_11_lc_fixed_input_summary.csv` 已包含 LC2010 null / recorded / DSSAT auto 的完整输出。因此本轮无需重跑 DSSAT，也无需训练，只需补齐合并表。

## 目标

生成 LC2010 完整对照表：

- Null
- Recorded/farmer practice
- DSSAT auto
- DQN best checkpoint（来自 LC2010 seed0 checkpoint summary，优先选择 total_reward 最大的 checkpoint）
- Official extension expert fixed DAP（来自 018_03）

并更新 018_03 的五站点 clean comparison，使 LC 不再缺字段。

## 严格限制

- 不训练。
- 不重跑 DSSAT。
- 不修改原始输入文件。
- 不覆盖旧结果；如需更新 018_03 clean 表，先读取原始数据并重建 clean 表。
- 所有过程记录到中文 MD。

## 输出

- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_04_lc2010_complete_comparison.csv`
- 更新后的 `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv`
- `docs/2026-07-09_018_04_lc2010_complete_four_scenario_merge_record.md`

