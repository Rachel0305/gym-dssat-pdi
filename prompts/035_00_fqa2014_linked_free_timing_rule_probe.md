# 035_00 FQA2014 linked 自由时序规则探针 prompt

## 背景

034_02–034_03 已确认旧 RL 外部动作可能被 `IRRIG=R/FERTI=R` 阻断；动态 RL 环境已修为 `IRRIG=L/FERTI=L`，且五站点 smoke 通过。

034_05 在 linked 修复后训练 MaskablePPO 50K，动作真实进入 DSSAT，但 FQA2014 结果不优：

- PPO 50K: yield 8080.13 kg/ha, I150/N240, WP_ET 2.22, PFP_N 33.7。
- official expert: yield 8318.39 kg/ha, I23/N82, WP_ET 2.32, PFP_N 101.4。

因此当前问题不是“动作是否生效”，而是 linked 真实执行后，自由时序 reward/约束是否能产生合理策略。

## 目标

在不训练 PPO/DQN 的情况下，先用一组固定规则策略探查 FQA2014 的自由时序动作空间：

1. 当前动作网格/安全层/linked 管理下，是否存在明显优于 034_05 PPO 50K 的规则策略；
2. 不同时机/总量策略对产量、WP_ET、PFP_N、simple_profit 的影响；
3. 当前 reward 是否把高投入 early dump 策略排得过高；
4. 为后续 linked 真实训练的 PPO/DQN 提供目标行为和诊断基准。

## 固定条件

- 站点年份：FQA2014。
- 不训练模型。
- 使用 034_04/034_05 同一套 linked 自由时序环境。
- DSSAT treatment 1 必须是 `IRRIG=L, FERTI=L`。
- 动作网格：
  - irrigation: `[0, 15, 30, 45]` mm/event
  - nitrogen: `[0, 40, 80, 120]` kg/ha/event
- 安全约束沿用当前配置：
  - 单季灌溉上限 160 mm
  - 单季施氮上限 250 kg/ha
  - 灌溉最小间隔 7 天
  - 施肥最小间隔 7 天
  - 灌溉 DAP 1–120
  - 施氮 DAP 1–90
- 奖励沿用当前 stress-aware reward，仅用于排序诊断，不作为唯一成功标准。

## 待测规则

- `noop`: 不灌溉、不施肥。
- `early_dump_cap`: 前期快速打满接近上限。
- `ppo_03405_replay`: 回放 034_05 PPO 50K 的确定性动作序列。
- `split_moderate_n160`: 分散式中低氮策略。
- `critical_i90_n200`: 关键期中等投入策略。
- `water_saving_n160`: 低水中氮策略。
- `delayed_late`: 故意推迟到中后期。
- `stress_triggered`: 只在状态胁迫指标超过阈值时行动。

## 验收标准

本任务只做诊断，不判定训练成功。

必须输出：

1. 每个规则的 daily CSV；
2. 每个规则的 DSSAT snapshot；
3. 每个规则的 safe action 总量与 `Summary.OUT` 实际 I/N 总量一致性；
4. 与 FQA2014 四情景基线和 034_05 PPO 50K 的同口径比较；
5. 中文实验记录 MD。

若所有规则都不如 expert，结论是当前动作空间/约束/reward 下尚未发现强候选，不允许据此宣称 PPO/DQN 无法成功。

若某些规则优于 034_05 PPO 50K 或接近/超过 expert，下一步应让 PPO/DQN 以该行为类型为诊断参照，而不是直接扩大全站点。

