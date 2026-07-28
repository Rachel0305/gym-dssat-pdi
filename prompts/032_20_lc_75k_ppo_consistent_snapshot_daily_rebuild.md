# 032_20 LC 75k PPO 一致 DSSAT snapshot 日过程重建与重绘

## 任务背景

032_17 已经生成了 LC2005-LC2020 的五情景日过程图，但 032_19 数据质量审计发现：PPO 候选日值表存在重复 date-DAP 行，且部分年份五情景降雨总量不一致；因此 032_17 的终值汇总仍可作为候选表现参考，但其中用于解释 PPO 决策合理性的 WSPD/NSTD 日过程图不应直接用于汇报。

## 本轮目标

只做审计和重新绘图，不训练、不改模型、不重选 checkpoint。

1. 使用已冻结的 LC 多年自由时序 stress-aware MaskablePPO 75k checkpoint，对 LC2005-LC2020 逐年做确定性评估，并保存 DSSAT snapshot。
2. 五情景日值全部尽量从一致的 DSSAT 输出解析：`Weather.OUT`、`PlantGro.OUT`、`SoilWat.OUT`、`MgmtEvent.OUT`。
3. 基线优先使用已有 snapshot：
   - `031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/...`
   - `031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/...`
   - `032_18_lc_missing_dssat_auto_daily_completion/snapshots/LCA/...`
   - 若 LC2010 的历史基线只有 027_05 snapshot，则明确标注来源；不静默混用。
4. 输出每年一张五情景日过程图，样式沿用已有套图：天气、土壤水、WSPD、NSTD、灌溉、施肥、籽粒/生物量、统一累计奖励。
5. 输出完整 QA 表：重复行、五情景降雨一致性、事件合计、snapshot 来源、缺失来源。

## 边界

- 不调用 `learn()`。
- 不改变 75k PPO 模型。
- 不修改原始 DSSAT 输入数据。
- 不覆盖 032_17；本轮另存到 `benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/`。
- 若某些历史基线没有 snapshot，本轮可以记录缺失或使用明确标注的历史日表兜底，但不得把兜底数据伪装成 snapshot。

## 成功判据

- LC2005-LC2020 每年至少生成 PPO 候选 snapshot-derived 日值。
- 对有五情景 snapshot 或可追溯兜底来源的年份，生成五情景图。
- 输出中文实验记录 MD，明确哪些年份完全由 snapshot 解析，哪些年份存在兜底来源。
- QA 中不得再出现 PPO 候选重复 date-DAP 行；若仍出现，必须停止并记录原因。
