# 019_07 FQ2016 leaching-cost=20 5K 短训练

## 目标

在 019_06 的 500-step smoke 中，`leaching_cost=20` 暂时表现最好：

- 产量最高；
- `final_cleach/NLCC=0`；
- 但施氮仍为 300 kg/ha。

本轮目标不是证明最终策略优越性，而是做一个低成本 5K seed0 短训练，判断 `leaching_cost=20` 是否在更长一点的训练下仍能维持：

1. 较高产量；
2. 较低氮淋洗；
3. 不明显压低灌溉导致产量损失；
4. 训练链路稳定，不 OOM。

同时保留 `leaching_cost=0` 作为无淋洗惩罚对照。

## 实验设置

- 站点年份：FQ2016
- 算法：DQN
- seed：0
- timesteps：5000
- checkpoint：每 1000 steps 评估一次
- 对照：
  - `leaching_cost=0`
  - `leaching_cost=20`
- 其他奖励项、动作空间、预算、窗口保持与 019_05/019_06 一致。

## 执行要求

- 必须使用指定 Docker 容器和虚拟环境：
  - Docker：`b2fd6726c8c1`
  - Python：`/opt/gym_dssat_pdi/bin/python`
- 不做 seed1。
- 不做 50K/100K。
- 不修改核心 reward 结构。
- 保存模型、日值 CSV、checkpoint summary 和日志。
- 训练后汇总两个系数在各 checkpoint 下的：
  - GWAD/HWAM
  - CWAD
  - 灌溉总量
  - 施氮总量
  - `cleach/NLCC`
  - reward
  - 最大水分/氮胁迫

## 判断标准

如果 `leaching_cost=20` 在 5K 下相比 `leaching_cost=0` 能维持接近或更高产量，同时降低 `cleach/NLCC`，则可以作为下一步 seed1 稳定性候选。

如果 `leaching_cost=20` 明显降低产量或行为不稳定，则说明 500-step smoke 的好结果不可放大，应回到无淋洗惩罚或重新设计淋洗项尺度。
