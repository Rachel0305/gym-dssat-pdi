# 013_07 禹城 2014 linked DQN 5K 多 seed 稳定性验证

## 背景

013_05 证明动态动作必须使用 `IRRIG=L / FERTI=L` 才能进入 DSSAT。

013_06 在 YC2014 上用 500 step smoke 验证了 linked DQN 动作通道已经修复：DQN wrapper 动作总量与 `MgmtEvent.OUT` 实际事件总量一致，产量从 null 的 7825 kg/ha 提高到约 9418 kg/ha。

## 目标

本阶段不继续调 reward、不改动作空间、不换站点，只做一个低风险稳定性验证：

- YC2014
- linked management: `IRRIG=L / FERTI=L`
- DQN
- 5000 timesteps
- seed0 和 seed1

检验：500 step smoke 中出现的高产结果是否能在 5K 训练下跨 seed 复现。

## 设置

- 奖励：`delta_grnwt - 1.0 * irrigation - 5.0 * nitrogen`
- 水预算：`I <= 120 mm`
- 氮预算：`N <= 300 kg/ha`
- 单日上限：`I <= 30 mm`, `N <= 100 kg/ha`
- 最小操作间隔：7 天

## 情景

只跑两个动态 DQN 情景，基线沿用 013_06：

1. `dqn_linked_free_daily`
2. `dqn_linked_agronomic_window`

## 判定标准

1. 动作通道必须继续通过：
   - wrapper action total 与 `MgmtEvent.OUT` total 应一致或高度一致。
2. 产量不能退回 null：
   - 若接近 9418 kg/ha，说明 5K 训练下仍可复现高产。
3. seed 稳定性：
   - seed0、seed1 如果都接近 recorded/固定高投入产量，则这条 DQN linked 线值得继续扩展到其他年份/站点。
   - 如果 seed 分化明显，先记录为稳定性不足，不继续扩大训练。

## 算力控制

- 不跑 PPO。
- 不重复 null/recorded/dssat_auto。
- 每个 seed 单独输出文件夹，避免互相覆盖。

