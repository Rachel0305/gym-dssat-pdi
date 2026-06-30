# 011_08 HLA 2010 official reward budgeted daily PPO 5k seed1

## 背景

011_07 seed0 已完成 5000 timesteps：

- budgeted daily PPO 流程跑通；
- I=120 mm、N=150 kg/ha 预算约束生效；
- final yield = 7854 kg/ha；
- 但策略表现为季节早期快速用完预算：
  - DAP 1/8/15 施完 N150；
  - DAP 1/8/15/22 灌完 I120。

## 本轮目标

只换随机种子，运行 seed1。

目的：

判断“早期打满预算”是否是 seed0 偶然，还是当前 official reward + budget wrapper 的稳定倾向。

## 设置

与 011_07 完全一致，只改：

- seed: 1

保持：

- year: HLA 2010
- timesteps: 5000
- reward: official `references/rewards.py`, scalarized by sum
- management: `IRRIG=L, FERTI=L`
- wrapper: `BudgetedDailyActionWrapper`
- budget:
  - I_total <= 120 mm
  - N_total <= 150 kg/ha
  - I_day <= 30 mm
  - N_day <= 50 kg/ha
  - min interval = 7 days
- PPO:
  - `n_steps=5`
  - `batch_size=5`
  - `n_epochs=1`
  - `gamma=0.99`

## 输出

目录：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/seed1_5000steps`

重点比较：

- 是否完成；
- I/N 总量；
- 管理事件 DAP；
- final yield；
- 与 seed0 的管理时序是否一致。

## Result

Seed1 completed successfully.

Comparison:

| seed | irrigation | nitrogen | fertilizer events | irrigation events | HWAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 150 | 3 | 4 | 7854 |
| 1 | 120 | 150 | 3 | 7 | 7854 |

Seed1 management:

| DAP | N | I |
| ---: | ---: | ---: |
| 2 | 50 | 20.6 |
| 9 | 50 | 14.2 |
| 16 | 50 | 13.9 |
| 23 | 0 | 15.0 |
| 30 | 0 | 19.2 |
| 37 | 0 | 19.1 |
| 44 | 0 | 18.0 |

Interpretation: early N saturation is stable across seeds. Irrigation is also early but the distribution varies by seed.
