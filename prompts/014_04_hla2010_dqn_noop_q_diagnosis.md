# 014_04 HLA2010 DQN no-op 退化 Q 值诊断

## 背景

014_03 中，HLA2010 DQN 统一流程复核出现了明显矛盾：

- 200 step smoke test 能产生有效操作，并达到约 7854 kg/ha；
- 但 free_daily seed0 5K 训练后，确定性评估完全不操作，产量退化到 null。

因此不能继续盲目跑 seed1，需要先低成本诊断 5K 模型为什么选择 no-op。

## 目标

不重新训练，只读取已有模型，检查 DQN 在 HLA2010 评估轨迹上的 Q 值：

- 比较 `free_daily_seed0_200steps` 和 `free_daily_seed0_5000steps`；
- 每个 DAP 输出四个离散动作的 Q 值；
- 记录模型实际选择的动作、safe action、奖励、产量和胁迫；
- 判断 5K no-op 是因为：
  1. action 0 的 Q 值明显最高；
  2. 四个动作 Q 值非常接近，策略没有学清楚；
  3. 某些早期动作被错误估值，导致后续永远不操作。

## 输入

- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03/2010/free_daily_seed0_200steps/`
- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03/2010/free_daily_seed0_5000steps/`

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_noop_q_diagnosis_014_04/014_04_hla2010_dqn_q_values_daily.csv`
- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_noop_q_diagnosis_014_04/014_04_hla2010_dqn_q_summary.csv`
- `docs/2026-06-30_014_04_hla2010_dqn_noop_q_diagnosis_record.md`

## 运行原则

- 不训练；
- 不改旧模型；
- 不运行长任务；
- 只做模型评估与 Q 值读取。
