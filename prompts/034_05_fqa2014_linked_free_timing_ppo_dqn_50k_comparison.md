# 034_05 FQA2014 linked 修复后自由时序 PPO/DQN 50K 对比 prompt

## 目标

在 034_03/034_04 已确认动态 RL 动作真实进入 DSSAT 后，进行一个小规模但较正式的 FQA2014 算法对比：

- MaskablePPO 50,000 timesteps；
- DQN 50,000 timesteps；
- 与 034_00 已统一输入链重跑的四情景基线比较。

## 固定边界

- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`。
- 站点年份：FQA2014。
- seed：0。
- 不做 checkpoint selection，只评估最终模型。
- 不扩展到全站点。
- 不调 reward、不调动作空间、不调约束。
- 不覆盖 033/034 旧结果。

## 训练/动作/奖励设置

沿用 032_00/033_04/034_04 的自由时序 stress-aware 设置：

- 动作网格：灌溉 `[0,15,30,45]` mm × 施氮 `[0,40,80,120]` kg/ha。
- 季节软上限：I ≤ 160 mm, N ≤ 250 kg/ha。
- 单次上限：I ≤ 45 mm, N ≤ 120 kg/ha。
- 最小操作间隔：灌溉 7 天，施氮 7 天。
- 灌溉允许 DAP 1–120；施氮允许 DAP 1–90。
- reward：`0.158*harvest_grnwt - 1.1*I - 1.58*N + stress_relief_bonus`，整体乘 `0.001`。
- 动态 DSSAT 管理模式：`IRRIG=L, FERTI=L`。

## 必须检查

对 PPO/DQN：

1. 训练是否完成；
2. eval replay 是否完成；
3. safe-action 总量是否与 DSSAT `Summary.OUT` 的 IRCM/NICM 一致；
4. `OVERVIEW.OUT` 是否显示 linked management；
5. 输出日值表、模型、快照、summary。

## 比较指标

与 034_00 四情景基线同口径比较：

- grain yield；
- actual irrigation；
- actual nitrogen；
- ETCP；
- WP_ET；
- PFP_N；
- simple profit / reward-like score。

## 输出

- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_training_summary.csv`
- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_eval_summary.csv`
- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_scenario_comparison.csv`
- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/daily_outputs/FQA/*.csv`
- `docs/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison_record.md`
