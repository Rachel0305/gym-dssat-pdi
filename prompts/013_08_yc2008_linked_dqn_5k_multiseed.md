# 013_08 禹城 2008 linked DQN 5K 多 seed 扩展验证

## 背景

013_07 在 YC2014 上证明 `linked free daily` DQN 在 5K、seed0/seed1 下稳定达到高产，并且动作总量与 `MgmtEvent.OUT` 实际事件一致。

013_01 前向筛选显示 YC2008 也存在一定水氮管理响应：recorded 产量高于 null，且氮胁迫从 null 的较高水平下降。因此把 linked DQN 扩展到 YC2008。

## 目标

检验同一套 DQN linked free daily 设置是否能在另一个禹城年份复现：

- 动作通道有效；
- 产量高于 null；
- seed0/seed1 结果稳定。

## 设置

- 站点年份：YC2008
- 管理模式：`IRRIG=L / FERTI=L`
- 算法：DQN
- 步数：5000
- seed：0 和 1
- 奖励：`delta_grnwt - 1.0 * irrigation - 5.0 * nitrogen`
- 水预算：`I <= 120 mm`
- 氮预算：`N <= 300 kg/ha`
- 日上限：`I <= 30 mm`, `N <= 100 kg/ha`
- 最小操作间隔：7 天

## 情景

只跑两个动态 DQN 情景，基线沿用 013_01：

1. `dqn_linked_free_daily`
2. `dqn_linked_agronomic_window`

## 注意

封丘 2007/2010 在 013_01 当前筛选中 null 与 recorded 差异很小甚至 null 略高，暂不作为优先扩展年份；FQ2008 结果缺失，不能直接判定是否有优化空间。

