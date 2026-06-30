# 011_06 HLA 2010 official reward budgeted daily PPO smoke

## 背景

011_04/011_05 已经确认：

- HLA MZX Jinja 占位符修复成功；
- `mode='all'` 渲染为 `IRRIG=L, FERTI=L`；
- 日尺度 linked `env.step()` 本身很快；
- SB3 PPO smoke 已经跑通；
- 但未加约束时，PPO 初始策略会每天输出接近 action space 中点的水氮动作，导致极端过量投入。

## 本轮目标

保留日尺度交互，但加入最小安全约束 wrapper。

这不是正式训练，只是 smoke：

1. 验证约束 wrapper 能把日动作和季节总量限制在合理范围内；
2. 验证经过约束后的 action 仍能进入 DSSAT `MgmtEvent.OUT`；
3. 验证 SB3 PPO 训练流程仍能跑通；
4. 不解读产量优劣。

## 约束设置

初版最小约束：

- seasonal irrigation budget: `I_total <= 120 mm`
- seasonal nitrogen budget: `N_total <= 150 kg/ha`
- daily irrigation cap: `I_day <= 30 mm`
- daily nitrogen cap: `N_day <= 50 kg/ha`
- operation interval: `min_interval_days = 7`

解释：

- PPO 仍然每天接收状态；
- PPO 仍然每天输出 action；
- wrapper 将 action 转成安全 action；
- 如果距离上次实际操作不足 7 天，则该日 action 强制为 0；
- 如果剩余预算不足，则 action 截断到剩余预算。

## 运行设置

- year: HLA 2010
- timesteps: 50
- seed: 0
- reward: official `references/rewards.py`, scalarized by sum
- PPO smoke hyperparameters:
  - `n_steps=5`
  - `batch_size=5`
  - `n_epochs=1`

## 判断标准

通过 smoke 需要：

- 训练完成；
- eval 完成；
- eval CSV 包含 raw action 和 safe action；
- `safe_irrigation_total <= 120`
- `safe_n_total <= 150`
- `MgmtEvent.OUT` 中实际管理量不超过预算；
- 无 OOM、无残留进程。

## 注意

50 steps 仍然只是 smoke，不代表策略收敛。

## Result

50-step smoke completed successfully.

Observed eval totals:

- safe irrigation total in CSV: 120 mm
- safe nitrogen total in CSV: 150 kg/ha
- MgmtEvent irrigation total: 120.1 mm, due to one-decimal event rounding
- MgmtEvent fertilizer total: 150 kg/ha
- MgmtEvent fertilizer events: 3
- MgmtEvent irrigation events: 5
- final harvest yield in budgeted eval block: 7854 kg/ha

This validates the budgeted daily PPO pipeline, but not policy convergence.
