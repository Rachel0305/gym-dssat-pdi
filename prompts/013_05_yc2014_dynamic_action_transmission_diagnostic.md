# 013_05 YC2014 dynamic action transmission diagnostic

## 目标

诊断 DQN/PPO 动态动作为什么没有进入 DSSAT `MgmtEvent.OUT`。

本实验不训练模型，只用人工 forced action 测试动作链路：

`real action -> normalized action -> GymDssatWrapper -> PDI/DSSAT -> MgmtEvent.OUT -> PlantGro.OUT`

## 核心假设

如果 `.MZX` 的 management line 是 `IRRIG=N, FERTI=N`，PDI step 动作会被 wrapper 记录，但不会进入 DSSAT 管理事件。

如果 `.MZX` 的 management line 是 `IRRIG=L, FERTI=L`，PDI step 动作应进入 `MgmtEvent.OUT`，并改变 `PlantGro.OUT`。

## 输入

- 站点：YC
- 年份：2014
- 输入包：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC`
- treatment：`TRNO=2`
- 容器：`b2fd6726c8c1`
- Python：`/opt/gym_dssat_pdi/bin/python`

## 场景

1. `static_null_zero_action`
   - management: `N/N`
   - action: all zero

2. `static_forced_action`
   - management: `N/N`
   - forced action: DAP 43 N=100, I=30; DAP 50 N=100, I=30; DAP 57 N=100, I=30

3. `linked_null_zero_action`
   - management: `L/L`
   - action: all zero

4. `linked_forced_action`
   - management: `L/L`
   - forced action: DAP 43 N=100, I=30; DAP 50 N=100, I=30; DAP 57 N=100, I=30

## 判读

- 如果 `static_forced_action` 的 action total > 0，但 `MgmtEvent` = 0，说明 N/N 会屏蔽动态动作；
- 如果 `linked_forced_action` 的 `MgmtEvent` > 0，且产量/胁迫变化，说明 gym/PDI 动态通道本身可用；
- 后续 DQN/PPO 训练必须使用 `IRRIG=L, FERTI=L`，不能再用 `null` 的 `N/N` 输入。
