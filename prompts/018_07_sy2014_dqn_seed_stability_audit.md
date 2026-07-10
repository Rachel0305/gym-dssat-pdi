# 018_07 SY2014 DQN 跨 seed 稳定性复核

## 背景

018_05 审计显示，SY2014 是当前非常强的代表年份之一：

- DQN 产量高于 official extension expert
- DQN 比 official extension expert 节水
- 与 official extension expert 施氮量相当

但目前这些判断主要基于单一代表 DQN 结果，尚未明确跨 seed 是否稳定。

## 目标

优先读取现有 SY2014 结果，判断是否已经具备跨 seed 证据：

1. 是否已有 seed0 / seed1 的 checkpoint 或最终汇总结果；
2. 若已有，直接做稳定性审计；
3. 若没有，只记录缺口，不立即训练；
4. 输出结论：SY2014 能否称为跨 seed 稳定成功案例，还是仍然只是 promising candidate。

## 输入优先级

优先检查：

- `DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/`
- `DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/`
- 其他已存在的 SY 结果目录

## 限制

- 不训练。
- 不重跑 DSSAT。
- 不改奖励函数。
- 不改原始输入文件。
- 只做已有结果审计。

## 输出

写入：

`DSSAT_auto_validation/extension_expert_baseline_018_03/018_07_sy2014_seed_stability_audit/`

至少包括：

- `018_07_sy2014_seed_inventory.csv`
- `018_07_sy2014_seed_stability_summary.csv`
- `docs/2026-07-09_018_07_sy2014_dqn_seed_stability_audit_record.md`

如果没有足够 seed 结果，要明确写：

- 目前已有哪个 seed
- 缺哪个 seed
- 下一步如果要补，最小补充实验是什么

