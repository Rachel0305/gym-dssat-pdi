# 013_06 禹城 2014 linked DQN smoke

## 背景

013_03 和 013_04 发现 DQN wrapper 记录了灌溉/施氮动作，但 `MgmtEvent.OUT` 里没有对应事件，产量仍等于 null。013_05 进一步证明：当输入模板为 `IRRIG=N / FERTI=N` 时，动态动作会被 DSSAT 屏蔽；当输入模板为 `IRRIG=L / FERTI=L` 时，同样动作能进入 `MgmtEvent.OUT` 并改变产量。

## 目标

在不做大训练的前提下，重新运行 YC2014 DQN smoke，唯一关键修复是：

- DQN 动态动作场景保留原始管理表和 MI/MF 指针；
- 只把管理模式设置为 `IRRIG=L, FERTI=L`；
- 检查 DQN 输出动作是否进入 DSSAT `MgmtEvent.OUT`。

## 运行设置

- 站点年份：YC2014
- 算法：DQN
- 步数：500 timesteps
- seed：0
- 奖励：`delta_grnwt - 1.0 * irrigation - 5.0 * nitrogen`
- 预算：`I <= 120 mm`, `N <= 300 kg/ha`
- 日上限：`I <= 30 mm`, `N <= 100 kg/ha`
- 最小操作间隔：7 天

## 情景

1. `null`
2. `recorded`
3. `dssat_auto`
4. `dqn_linked_free_daily`
5. `dqn_linked_agronomic_window`

## 判定标准

本阶段不是判断 DQN 最优性，只判断动作链路是否修复：

- `action_irrigation_total > 0` 时，`mgmt_event_irrigation_total` 也应大于 0；
- `action_fertilizer_total > 0` 时，`mgmt_event_fertilizer_total` 也应大于 0；
- DQN 产量不应继续完全等于 null；
- 若仍不一致，停止训练，继续诊断 action channel。

## 输出

- 汇总 CSV
- 日值 CSV
- 高对比过程图
- 中文实验记录 MD
