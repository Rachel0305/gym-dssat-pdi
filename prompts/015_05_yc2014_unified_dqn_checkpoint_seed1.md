# 015_05 YC2014 统一 DQN checkpoint selection 跨 seed 复核

## 背景

015_04 在 YC2014 seed0 上确认：

- 同一次 50K DQN 训练中，最终模型并不一定最好；
- 5K/10K/20K checkpoint 可以达到约 9417–9418 kg/ha；
- final checkpoint 产量下降到约 8939 kg/ha，且氮管理变差；
- 因此正式 DQN 训练应采用 checkpoint selection，而不是直接使用 final model。

## 目的

本轮只换随机种子，从 seed0 改为 seed1，其他设置完全不变，验证：

1. 统一 DQN 框架是否能跨 seed 复现高产水氮策略；
2. checkpoint selection 是否仍能找到优于 final model 的 checkpoint；
3. YC2014 是否可以作为正式 DQN 成功案例。

## 固定设置

- 站点年份：YC2014；
- 算法：DQN；
- seed：1；
- 总训练步数：50K；
- checkpoint 间隔：5K；
- 奖励：

```text
R_t = max(0, GRNWT_t - GRNWT_{t-1}) - 1.0 * I_t - 5.0 * N_t
```

- 动作空间：9 个离散动作：
  - I ∈ {0, 15, 30} mm；
  - N ∈ {0, 50, 100} kg/ha；
- 预算：
  - I ≤ 120 mm；
  - N ≤ 300 kg/ha；
- 最小操作间隔：7 days；
- 管理模式：IRRIG=L, FERTI=L。

## 输出

保存到：

```text
DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/
```

输出：

- 每个 checkpoint 的日值 CSV；
- checkpoint 汇总 CSV；
- checkpoint 诊断图；
- 中文实验记录 MD；
- 与 015_04 seed0 的简要对比。

## 判断标准

如果 seed1 的 best checkpoint 也能达到：

- 产量接近 9400 kg/ha；
- 灌溉接近 120 mm；
- 施氮不为 0，且在 200–300 kg/ha 左右；
- 胁迫水平合理；

则说明 YC2014 统一 DQN + checkpoint selection 具有初步跨 seed 可复现性。

如果 seed1 找不到高产 checkpoint，则说明 seed0 结果仍可能具有偶然性，暂不能作为正式成功案例。

