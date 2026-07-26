# 034_02 PPO 外部动作落地链路根因审计 prompt

## 背景

034_00/034_01 已经发现：当前自由时序 PPO wrapper 的 `safe_action_amir/safe_action_anfer` 有非零动作，但 DSSAT `Summary.OUT` 中 `IRCM/NICM` 仍为 0，且 `ppo_external_replay` 与 `null_external_noop` 作物结果一致。

本任务只追问一个更小的问题：

> 当前外部动作没有进入 DSSAT，是否由渲染后的 treatment 管理模式仍为 `IRRIG=R, FERTI=R` 导致？

旧的 linked DQN 记录显示，动态 RL 动作需要把对应 treatment 管理模式切到 `IRRIG=L, FERTI=L`，同时保留原始管理表和指针。

## 固定边界

- 不训练 PPO/DQN。
- 不修改原始输入目录。
- 不修改已有 033/034 结果。
- 只在新的 `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/` 下生成渲染副本、快照和 CSV。
- 使用 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`。
- 使用 FQA2014 做最小复核，动作固定为 DAP1 外部动作 `I45/N80`，之后 no-op。

## 审计设计

1. 解析 034_01 现有 `ppo_external_replay` 快照：
   - treatment 行 `IC/MI/MF`；
   - management 行 `IRRIG/FERTI`；
   - `OVERVIEW.OUT` 的 `MANAGEMENT OPT`。
2. 运行两个最小外部动作环境：
   - `external_rr`: 渲染后保持当前 `IRRIG=R, FERTI=R`，DAP1 发 I45/N80；
   - `external_ll`: 渲染后仅把 treatment 1 改为 `IRRIG=L, FERTI=L`，DAP1 发同一个 I45/N80。
3. 比较两组：
   - Python 层 safe-action 总量；
   - DSSAT `Summary.OUT` 的实际 IRCM/NICM；
   - 产量；
   - `OVERVIEW.OUT` 的管理模式。

## 判定

- 若 `external_rr` 的 safe-action > 0 但 Summary I/N = 0，而 `external_ll` 的 Summary I/N > 0，则判定：当前 PPO 外部动作未生效的直接根因是 management mode 没有切到 linked。
- 若 `external_ll` 仍为 0，则继续检查 gym-DSSAT/PDI 动作字段映射，而不能直接归因于 management mode。

## 输出

- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_mode_parse.csv`
- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_forced_action_summary.csv`
- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_forced_action_daily.csv`
- `docs/034_02_ppo_external_action_linkage_rootcause_audit_record.md`
