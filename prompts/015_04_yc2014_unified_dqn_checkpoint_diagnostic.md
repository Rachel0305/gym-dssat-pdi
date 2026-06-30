# 015_04 YC2014 统一 DQN checkpoint 诊断

## 背景

015_03 发现 YC2014 统一 DQN 在相同 reward 和 seed 下，训练步长不同会得到很不稳定的策略：

- 5K：I120/N250，产量约 9418 kg/ha；
- 10K：I15/N300，产量约 8749 kg/ha；
- 20K：I120/N100，产量约 8939 kg/ha；
- 50K：I90/N0，产量约 8056 kg/ha。

这说明问题可能不是奖励函数本身，而是 DQN 训练过程中存在策略漂移；最终模型不一定是最优模型。

## 目的

在同一次 50K DQN 训练过程中，每隔 5K 保存并评估 checkpoint，判断：

1. 最佳策略是否出现在训练中途；
2. final model 是否明显差于 best checkpoint；
3. 后续正式训练是否应该采用 best checkpoint，而不是最终模型。

## 固定设置

- 站点年份：YC2014；
- 算法：DQN；
- seed：0；
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
DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/
```

输出：

- 每个 checkpoint 的日值 CSV；
- checkpoint 汇总 CSV；
- checkpoint 诊断图；
- 中文实验记录 MD；
- 记录 best checkpoint 的判断依据。

## 判断规则

如果某个中途 checkpoint 的产量、reward、水氮操作明显优于 50K final，则说明正式 DQN 训练不能简单使用最终模型，必须引入 checkpoint selection。

如果所有 checkpoint 都不稳定，则说明需要继续改 DQN 训练机制。

如果 final checkpoint 也是最优，则说明 015_01 的异常可能来自配置差异或偶然运行，需要回查脚本。

