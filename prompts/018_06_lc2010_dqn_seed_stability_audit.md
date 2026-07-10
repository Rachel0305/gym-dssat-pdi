# 018_06 LC2010 DQN 跨 seed 稳定性复核

## 背景

018_04 已补齐 LC2010 五情景对照表。LC2010 seed0 的 DQN best checkpoint 表现很好：

- 产量约 8739 kg/ha
- 灌溉约 90 mm
- 施氮 0 kg/ha
- 与 DSSAT auto / official extension expert 产量相当，但水氮投入更少

但此前检查发现 seed1 虽然产量也接近 8739 kg/ha，却可能使用 120 mm 灌溉和 300 kg/ha 氮，说明策略资源效率可能不稳定。

## 目标

不重新训练，只读取已有 LC2010 DQN seed0 / seed1 checkpoint summary，判断：

1. 两个 seed 是否都能达到高产平台；
2. 两个 seed 是否都能节水；
3. 两个 seed 是否都能节氮；
4. 是否可以把 LC2010 作为“跨 seed 稳定成功案例”；
5. 如果不能，下一步应该补训练、改奖励，还是暂时作为“单 seed 成功候选”。

## 输入

读取：

- `DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed0_5000steps/checkpoint_summary.csv`
- `DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed1_5000steps/checkpoint_summary.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_04_lc2010_complete_comparison.csv`

## 限制

- 不训练。
- 不重跑 DSSAT。
- 不改奖励函数。
- 不改原始输入文件。
- 只做现有结果审计。

## 输出

写入：

`DSSAT_auto_validation/extension_expert_baseline_018_03/018_06_lc2010_seed_stability_audit/`

至少包括：

- `018_06_lc2010_seed_checkpoint_table.csv`
- `018_06_lc2010_seed_best_summary.csv`
- `figures/018_06_lc2010_seed_stability.png`
- `docs/2026-07-09_018_06_lc2010_dqn_seed_stability_audit_record.md`

## 判定口径

LC2010 只有在以下条件同时满足时，才能称为“跨 seed 稳定成功案例”：

1. seed0 和 seed1 的最佳 checkpoint 均追平/超过 DSSAT auto 和 official extension expert 产量；
2. seed0 和 seed1 至少都比 official extension expert 节水；
3. seed0 和 seed1 至少都不明显多于 official extension expert 的施氮量；
4. 若一个 seed 高效、另一个 seed 靠高投入追平产量，则只能称为“产量稳定、资源效率不稳定”。

