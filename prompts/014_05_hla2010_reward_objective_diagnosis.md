# 014_05 HLA2010 奖励目标诊断：no-op 是否是当前经济奖励下的理性结果

## 背景

014_03 中 HLA2010 DQN 出现：

- 200 step smoke：DQN 会灌溉/施肥，产量约 7854 kg/ha；
- 5K：DQN 确定性评估全程 no-op，产量约 6956 kg/ha；
- 014_04 Q 值诊断显示 5K 模型每天都选择 action 0，说明不是动作链路坏，而是模型学到 no-op 价值最高。

现在需要继续诊断：这是否其实是当前奖励函数本身导致的合理结果。

当前奖励：

```text
reward = ΔGRNWT - 1.0 × irrigation - 5.0 × nitrogen
```

## 目标

不训练、不运行 DSSAT，只读取已有 HLA2010 轨迹，重算不同奖励口径：

1. 当前 undiscounted economic return；
2. 相对 null 的产量增益、灌溉成本、氮成本；
3. 在当前 water_cost=1.0 下，计算 DQN 操作轨迹要优于 no-op 所需的氮成本上限；
4. 判断 HLA2010 no-op 是训练失败，还是当前经济目标下的合理最优。

## 输入

- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03/2010/free_daily_seed0_200steps/dqn_eval_daily.csv`
- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03/2010/free_daily_seed0_5000steps/dqn_eval_daily.csv`
- `DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03/2010/agronomic_window_seed0_200steps/dqn_eval_daily.csv`
- `DSSAT_auto_validation/HLA_2004/hla_2010_2015_four_scenario_with_ppo/hla_2010_2015_four_scenario_summary.csv`

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_reward_objective_diagnosis_014_05/014_05_hla2010_reward_rescore_summary.csv`
- `docs/2026-06-30_014_05_hla2010_reward_objective_diagnosis_record.md`

## 原则

- 不训练；
- 不改奖励函数；
- 先把当前奖励函数的逻辑后果讲清楚。
