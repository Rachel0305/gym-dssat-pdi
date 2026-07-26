# 034_01 PPO 动作是否真实进入 DSSAT 的三角审计

## 背景

034_00 在重建统一输入链、IC=1 的四情景基线时发现：通过 `env.step(action)` 向 DSSAT 发送外部灌溉/施氮动作时，`Summary.OUT` 的 `IRCM/NICM` 与作物轨迹没有发生响应。固定 recorded/expert 基线因此改为写入渲染后的 `.MZX` 静态管理表执行。

033_04 PPO 结果中的灌溉量、施氮量来自 `safe_action_amir/safe_action_anfer` 的代码层累计值，而不是直接从 `Summary.OUT` 读取。因此在继续比较 PPO 与四情景基线之前，必须确认：

> PPO 的动作到底有没有真实进入 DSSAT 并改变作物模拟？

## 目标

对 033_04 已有的 MaskablePPO checkpoint 做代表性三角审计。每个站点选取一个已有验证年份、已有非零 PPO 动作的 checkpoint，比较三种回放：

1. `null_external_noop`：同一输入下，外部 step 全部 no-op；
2. `ppo_external_replay`：加载 033_04 PPO checkpoint，经原 PPO wrapper/action safety 路径进行确定性回放；
3. `ppo_static_mzx_same_schedule`：先从 `ppo_external_replay` 读取 safe-action 序列，再把同一序列写入 `.MZX` 静态管理表，step 阶段只发送 no-op。

## 固定边界

- 不重新训练 PPO；
- 不调 reward；
- 不改动作空间；
- 不改 033_04 模型；
- 只使用 033_04 的已有 checkpoint 和可验证年份；
- 只使用 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`；
- 必须启用 `IC=1, MI=1, MF=1`；
- PPO 代码层动作总量与 DSSAT `Summary.OUT` 实际执行总量必须分开报告。

## 判定逻辑

若出现：

```text
ppo_external_replay 的 safe-action 总量 > 0
但 ppo_external_replay 的 Summary.OUT IRCM/NICM ≈ null_external_noop
同时 ppo_static_mzx_same_schedule 的 Summary.OUT IRCM/NICM > 0 且轨迹/产量改变
```

则说明：PPO 产生了动作，安全层也记录了动作，但外部 action 通道没有让 DSSAT 实际执行这些动作。

若 `ppo_external_replay` 与 `ppo_static_mzx_same_schedule` 的 `Summary.OUT` 和轨迹一致，则说明 PPO 外部 action 通道有效。

若结果混合，则停止并记录，不扩大实验。

## 输出

结果目录：

```text
benchmark_results/034_01_ppo_action_dssat_effect_audit/
```

主要文件：

```text
evaluation/034_01_case_selection.csv
evaluation/034_01_summary.csv
evaluation/034_01_daily.csv
evaluation/034_01_comparison.csv
docs/034_01_ppo_action_dssat_effect_audit_record.md
```

## 注意

本任务只回答 PPO 动作是否真实落地到 DSSAT，不评价 PPO 是否优于四情景，不做新训练。
