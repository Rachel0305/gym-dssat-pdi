# 011_07 HLA 2010 official reward budgeted daily PPO 5k

## 背景

011_06 已经验证：

- HLA 2010 Jinja linked management 正常；
- official reward 可以用于 SB3；
- daily PPO 流程跑通；
- `BudgetedDailyActionWrapper` 能把动作限制在：
  - seasonal irrigation <= 120 mm
  - seasonal nitrogen <= 150 kg/ha
  - daily irrigation <= 30 mm
  - daily nitrogen <= 50 kg/ha
  - min interval = 7 days
- 50-step smoke 成功，eval 中 DSSAT MgmtEvent 与 CSV 累计基本一致。

## 本轮目标

在相同设置下直接运行 5000 timesteps，初步观察 PPO 是否能在预算约束下形成可用日尺度策略。

## 设置

- year: HLA 2010
- seed: 0
- timesteps: 5000
- reward: official `references/rewards.py`, scalarized by sum
- management: `IRRIG=L, FERTI=L`
- wrapper: `BudgetedDailyActionWrapper`
- PPO smoke/training hyperparameters:
  - `n_steps=5`
  - `batch_size=5`
  - `n_epochs=1`
  - `gamma=0.99`

## 输出

目录：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/seed0_5000steps`

保存：

- model
- debug log
- eval daily CSV
- `pdi_tmp_snapshot_eval`
- `event_summary.json`

## 判断重点

这次可以初步看行为，但仍然不是最终结论。

重点检查：

1. 是否完成 5000 steps；
2. 是否无 OOM、无残留进程；
3. eval 总水氮是否仍在预算内；
4. MgmtEvent 是否与 CSV 一致；
5. 产量是否高于 null；
6. 动作时间是否比 50-step smoke 更合理。

## Result

5k completed successfully.

Key outputs:

- no timeout;
- no residual process;
- CSV safe irrigation total: 120 mm;
- CSV safe nitrogen total: 150 kg/ha;
- MgmtEvent irrigation total: 120.0 mm;
- MgmtEvent fertilizer total: 150 kg/ha;
- final budgeted PPO eval harvest yield: 7854 kg/ha.

Actual eval management:

| DAP | Fertilizer | Irrigation |
| ---: | ---: | ---: |
| 1 | 50 | 30 |
| 8 | 50 | 30 |
| 15 | 50 | 30 |
| 22 | 0 | 30 |

Interpretation: the pipeline works, but the learned/evaluated policy is still early-season budget saturation rather than a refined agronomic schedule.
