# 017_06 FQ2019 过程图与 seed1 稳定性验证

## 目的

017_05 显示 FQ2019 seed0 在 checkpoint 10000 达到较高产量，但使用 I120/N300。  
本实验做两个判断：

1. 绘制 FQ2019 checkpoint10000 四情景过程图，检查灌溉/施肥时机是否合理；
2. 运行 FQ2019 seed1 训练，判断 seed0 的成功是否可跨 seed 复现。

## 情景

- null_zero
- recorded_shifted
- dssat_auto
- fq2019_dqn_seed0_ckpt10000

## 输入

- 基线日值/事件/summary：017_03 FQ 全年份四情景基线表；
- DQN seed0 checkpoint10000：017_05 `seed0_50000steps`；
- seed1 训练：使用 017_05 同一奖励函数、约束、动作窗口和训练参数。

## 输出

- FQ2019 四情景过程图；
- FQ2019 四情景日值、事件、summary；
- FQ2019 seed1 checkpoint summary；
- 实验记录。

