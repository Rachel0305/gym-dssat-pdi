# 034_04 linked 修复后自由时序 PPO/DQN 训练 smoke prompt

## 目标

在 034_03 已确认动态 RL 外部动作可通过 `IRRIG=L, FERTI=L` 进入 DSSAT 后，做一次最小训练闭环：

> 训练 PPO/DQN → 确定性回放 → 检查 Python safe-action 与 DSSAT Summary.OUT 实际执行 I/N 是否一致。

## 固定边界

- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`。
- 站点年份：FQA2014。
- seed：0。
- 算法：MaskablePPO 与 DQN。
- 训练步数：5,000 timesteps。
- 不做 checkpoint selection，只评估最终模型。
- 不扩展到全站点，不下最终性能结论。
- 不覆盖 033/034 旧结果。

## 训练/动作/奖励设置

沿用 032_00/033_04 的自由时序 stress-aware 设置：

- 动作网格：灌溉 `[0,15,30,45]` mm × 施氮 `[0,40,80,120]` kg/ha。
- 季节软上限：I ≤ 160 mm, N ≤ 250 kg/ha。
- 单次上限：I ≤ 45 mm, N ≤ 120 kg/ha。
- 最小操作间隔：灌溉 7 天，施氮 7 天。
- 灌溉允许 DAP 1–120；施氮允许 DAP 1–90。
- reward：`0.158*harvest_grnwt - 1.1*I - 1.58*N + stress_relief_bonus`，整体乘 `0.001`。

## 必须检查

对每个算法：

1. 模型是否训练完成；
2. eval replay 中 Python safe-action 总量；
3. DSSAT `Summary.OUT` 中实际 IRCM/NICM；
4. safe-action 与 Summary.OUT 是否一致；
5. `OVERVIEW.OUT` 是否为 `IRRIG=L, FERT=L`；
6. 输出日值表、快照、训练/评估 summary。

## 通过标准

本 smoke 的通过标准不是“产量优于基线”，而是接口闭环：

- 训练成功；
- replay 完成；
- 若 safe-action 总量 > 0，则 Summary.OUT 实际 I/N 必须与 safe-action 一致；
- `OVERVIEW.OUT` 必须显示 linked management。

若 PPO/DQN 都通过闭环，才允许后续进入更长训练或全站点重跑。

## 输出

- `benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/evaluation/034_04_training_summary.csv`
- `benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/evaluation/034_04_eval_summary.csv`
- `benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/daily_outputs/FQA/*.csv`
- `docs/034_04_linked_free_timing_ppo_dqn_train_smoke_record.md`
