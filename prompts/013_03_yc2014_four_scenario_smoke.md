# 013_03 YC2014 四情景 smoke

## 目标

在 `YC 2014` 新参数/新输入包上，先做一个低算力四情景 smoke，不做长训练，只回答三个问题：

1. 四个情景能否全部顺利跑通；
2. RL 情景的动作、日志、日值、图表链路是否正常；
3. 在当前 `YC 2014` 响应面下，RL 是否至少表现出“会管理”的基本行为。

## 四情景

1. `null`
2. `recorded`
3. `dssat_auto`
4. `dqn_economic_smoke`

## RL 情景约定

- 算法：DQN
- 奖励：economic reward  
  `reward = delta_grnwt - water_cost * irrigation - nitrogen_cost * nitrogen`
- 成本系数：
  - `water_cost = 1.0`
  - `nitrogen_cost = 5.0`
- 预算：
  - `I <= 120 mm`
  - `N <= 300 kg/ha`
- 训练步数：`500`
- seed：`0`

## 动作设计

保持离散、低复杂度：

- `0`: do nothing
- `1`: irrigation only
- `2`: nitrogen only
- `3`: irrigation + nitrogen

但允许单次氮动作高于旧版 50 kg/ha，以适配 `YC 2014` 需要较高氮投入的响应面。

## 输出

目录：

`DSSAT_auto_validation/multisite_new_cultivar_yc2014_four_scenario_smoke_013_03/`

至少包含：

- `013_03_yc2014_four_scenario_daily.csv`
- `013_03_yc2014_four_scenario_summary.csv`
- 四情景过程图
- 中文实验记录

## 成功判据

- 不是看它是否立刻超过 recorded；
- 先看它是否：
  - 没有明显无意义的连续乱打动作；
  - 能在有胁迫的窗口附近出现管理动作；
  - 最终投入与产量量级合理；
  - 图和日值能支持后续正式实验。
