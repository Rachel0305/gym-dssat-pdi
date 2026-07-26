# 034_03 动态 RL linked management 修复 smoke prompt

## 目标

验证 034_02 发现的根因修复是否生效：动态 RL 环境创建时自动把当前 treatment 管理模式切为 `IRRIG=L, FERTI=L`，使 PPO/DQN 外部动作真正进入 DSSAT。

## 固定边界

- 不重新训练。
- 不修改原始输入目录。
- 不覆盖 033_04、034_01、034_02 结果。
- 只复用 033_04 已有 MaskablePPO checkpoint。
- 每个站点选 034_01 同样的一个 2014 checkpoint 案例。
- 对每个站点跑两种动态外部回放：
  - `null_dynamic_linked_noop`：linked 模式下全 no-op；
  - `ppo_dynamic_linked_replay`：linked 模式下加载 PPO checkpoint 确定性回放。

## 通过标准

对每个站点：

1. `null_dynamic_linked_noop` 的 DSSAT `Summary.OUT` 中 IRCM/NICM 为 0；
2. `ppo_dynamic_linked_replay` 的 Python safe-action 总量 > 0；
3. `ppo_dynamic_linked_replay` 的 DSSAT `Summary.OUT` 中 IRCM/NICM 与 safe-action 总量一致；
4. `OVERVIEW.OUT` 显示 `IRRIG=L, FERT=L`。

若 5/5 通过，则允许后续重跑自由 PPO/DQN 训练；否则继续停在接口修复阶段。

## 输出

- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_summary.csv`
- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_comparison.csv`
- `docs/034_03_dynamic_rl_linked_management_fix_smoke_record.md`
