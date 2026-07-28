# 037_07 静态 level-1 四基线全年份重建

## 背景

037_05 发现 034_00 的手工基线存在管理事件链路问题：`fileX.MZX` 中计划了多次灌溉/施肥，但 `DSSAT48.INP` 和 `MgmtEvent.OUT` 只执行了部分事件。037_06 用 linked-action 证明手工计划可以通过 step action 进入 DSSAT，但 recorded farmer 中存在单次灌溉超过动作通道上限的问题，不能用 linked-action 完整复现真实 recorded。

本任务修复静态 DSSAT 管理表：在 `@I IDATE` 和 `@F FDATE` 应用表中，同一个情景内的所有事件都使用同一个 management level 编号 `1`。旧脚本误把行首编号写成事件序号 `1,2,3...`，导致 DSSAT 只选择 level 1 的第一条事件。

## 目标

1. 不训练 PPO/DQN。
2. 使用 IC=1 的 multisite 输入源。
3. 对全部可用站点年份重建四基线：
   - null
   - recorded_farmer_template
   - official_extension_expert
   - dssat_auto
4. 对 recorded 和 expert 采用修复后的静态 DSSAT 管理表，允许复现超过 RL 动作通道单次上限的 recorded 事件。
5. 对每个情景输出 summary、daily、snapshot 和管理事件链路审计表。

## 通过标准

手工静态基线必须满足：

- 可执行季节内 planned irrigation/fertilizer 事件数与 `DSSAT48.INP` 一致；
- `DSSAT48.INP` 与 `MgmtEvent.OUT` 执行事件数一致；
- planned、INP、Mgmt、Summary 的季节总量在容差内一致；
- 播种后但收获后的计划事件单独标为 post_harvest，不作为失败。

`dssat_auto` 不按静态 planned 表审计，只记录 DSSAT 自动执行结果。

## 执行顺序

1. 先运行 smoke：每站点最早一个年份。
2. 若 smoke 中 manual_static_level1 基线链路通过，再运行 full。
3. 若 recorded 或 expert 仍有链路失败，停止，不使用该基线作为可信对照。

## 输出

- `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_*_baseline_summary.csv`
- `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_*_baseline_daily.csv`
- `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_*_management_event_audit.csv`
- `docs/037_07_static_level1_four_baseline_rebuild_record.md`

