# 013_04 YC2014 动作窗口对照

## 目标

验证“动作窗口”不是为了调参，而是一个可复核的动作空间约束实验。

本实验不直接追求 DQN 产量最优，而是比较：

1. 每天都允许操作的自由动作空间；
2. 基于实测管理时期和作物管理逻辑的农学窗口动作空间。

## 输入

- 站点：禹城 YC
- 年份：2014
- 输入包：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC`
- 容器：`b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`

## 情景

- `null`
- `recorded`
- `dssat_auto`
- `dqn_free_daily`
- `dqn_agronomic_window`

## DQN 固定设置

- timesteps = 500
- seed = 0
- reward = `delta_grnwt - water_cost * irrigation - nitrogen_cost * nitrogen`
- water_cost = 1.0
- nitrogen_cost = 5.0
- irrigation budget = 120 mm
- nitrogen budget = 300 kg/ha

## 动作窗口

### free_daily

- irrigation: DAP 1-120
- nitrogen: DAP 1-120

### agronomic_window

窗口不是根据 DQN 单次失败结果倒推，而是根据 YC2014 recorded 管理时点和固定扫描中的有效施用时段设置：

- irrigation: DAP 35-65
- nitrogen: DAP 1-10, DAP 35-55

## 输出

目录：

`DSSAT_auto_validation/multisite_new_cultivar_yc2014_action_window_comparison_013_04/`

包括：

- `013_04_yc2014_action_window_comparison_summary.csv`
- `013_04_yc2014_action_window_comparison_daily.csv`
- `figures/yc2014_action_window_comparison_process.png`
- `docs/2026-06-30_013_04_yc2014_action_window_comparison_record.md`

## 判读

如果 `agronomic_window` 的动作时机更合理，但产量仍不提升，说明问题不只是动作窗口，而可能还涉及奖励、训练量或 DQN 学习稳定性。

如果 `free_daily` 出现明显过早施肥，而 `agronomic_window` 减少这种行为，则动作窗口可以作为“可解释性约束”保留，但不能直接称为性能优化。
