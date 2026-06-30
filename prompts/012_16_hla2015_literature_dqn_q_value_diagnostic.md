# 012_16 HLA2015 文献对齐版 DQN Q-value 诊断

## 背景

012_15 参考文献 **A comparative study of deep reinforcement learning for crop production management** 改成了：

- 终端产量奖励 + 水氮投入成本；
- 25 个水氮离散组合动作；
- HLA2015、IC=1、窗口/预算约束不变。

012_15 seed0 5K 结果为：

- 产量约 7652 kg/ha；
- 总灌溉 102 mm；
- 总施氮 150 kg/ha；
- 不再退化为 null，但又把氮打满。

离线复评分显示，按同一套 012_15 reward，固定扫描中的 I60/N0 高于 DQN 学到的 I102/N150。因此问题更可能是 DQN Q-value 排序/信用分配没有学对，而不是 reward 公式本身偏好 N150。

## 目的

不重新训练，直接加载 012_15 已训练模型，检查：

1. 每个决策日 25 个动作的 Q-value 排序；
2. 灌溉窗口内，Q 最大动作是否偏向含氮动作；
3. 施氮窗口内，Q 是否错误高估高氮动作；
4. 最终策略为什么选择 I102/N150 而不是固定扫描中更优的 I60/N0。

## 执行要求

- 不重新训练；
- 使用指定 Docker 容器 `b2fd6726c8c1`；
- 使用指定虚拟环境 `/opt/gym_dssat_pdi/bin/python`；
- 加载模型：

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/2015/literature_aligned_seed0_5000steps/models/dqn_literature_aligned_probe.zip
```

- 输出：
  - 全生日逐日 Q-value CSV；
  - 关键 DAP 的 top actions CSV；
  - summary CSV/JSON；
  - 中文记录 MD。

## 判断重点

如果窗口内 Q 最大动作反复是含氮动作，而实际 reward 复评分不支持 N150，则说明：

- 012_15 的主要问题不是奖励目标本身；
- 而是 DQN 对高氮动作的长期价值估计有偏；
- 后续如果继续改算法，应优先考虑 n-step return / prioritized replay / 更直接的监督式 warm-start，而不是继续调奖励系数。
