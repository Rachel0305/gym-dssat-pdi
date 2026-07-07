# 017_03 FQ2016 seed1 checkpoint30000 跨年份迁移

## 目的

在不新增训练的前提下，使用 FQ2016 已验证表现较好的 DQN seed1 best-reward checkpoint 30000，迁移到封丘站所有可用年份，检查该模型在封丘站内部的跨年份泛化能力。

## 输入与控制变量

- 站点：FQ / Fengqiu
- 可用年份：使用 014_01 中已经完成 null / recorded_shifted / DSSAT auto 三个基线且结果有限的年份；继续排除旧记录中不可用的 FQ2007/FQ2008。
- DQN 模型：
  - `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/models/dqn_baseline_relative_checkpoint_30000.zip`
- DQN 框架：
  - 使用与 FQ2016 017_02 一致的 baseline-relative reward、离散动作预算 wrapper、FQ2016 seed1 checkpoint30000。
  - 不重新训练，只做 deterministic evaluation。
- 基线情景：
  - null_zero
  - recorded_shifted
  - dssat_auto
  - dqn_seed1_ckpt30000_transfer

## 节省算力原则

- 不重跑已经存在的 null / recorded_shifted / dssat_auto 基线，直接复用 014_01 的逐日输出、管理事件和 summary。
- 只新增 DQN 迁移 forward。
- 不做新训练、不调奖励函数、不覆盖旧结果。

## 输出

保存到：

`DSSAT_auto_validation/fq2016_seed1_checkpoint30000_all_year_transfer_017_03/`

需要输出：

- `fq2016_seed1_ckpt30000_transfer_daily.csv`
- `fq2016_seed1_ckpt30000_transfer_events.csv`
- `fq2016_seed1_ckpt30000_transfer_summary.csv`
- `fq2016_seed1_ckpt30000_transfer_success_by_year.csv`
- 每个年份一张四情景过程图：降雨、水分胁迫、氮胁迫、灌溉施肥措施、产量/生物量、累积奖励代理值。
- 跨年份汇总图。
- 实验记录 MD。

## 判读重点

1. DQN 是否能在多个年份中明显优于 null。
2. DQN 是否能接近或超过 recorded_shifted / dssat_auto。
3. DQN 是否比基线节水节氮。
4. 哪些年份属于成功迁移，哪些年份只是接近，哪些年份失败。

