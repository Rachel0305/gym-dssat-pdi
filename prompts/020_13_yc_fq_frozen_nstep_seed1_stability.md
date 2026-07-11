# 020_13 YC/FQ 冻结 n-step DQN seed1 稳定性复核

## 目的

在 020_12 已完成 YC2014、FQ2016 seed0 50K 的基础上，仅更换模型随机种子为 seed1，检验同一冻结框架的结果能否跨 seed 复现。

## 严格固定项

- 使用 `src/frozen_nstep_dqn_config_020_11.py` 作为唯一配置源。
- 奖励保持：每步 `-1.0*I - 5.0*N`；终止时增加 `max(0, GWAD_final-local_null_GWAD)`。
- 9 个离散动作、I120/N300 季节预算、单次 I30/N100、7 DAP 最短间隔、水氮窗口 DAP1–120 均不变。
- DQN 超参数不变：`n_steps=5`、learning rate、buffer、batch、gamma、探索率、target network 参数均保持冻结值。
- 各站点仍使用自己的同输入 null baseline；环境 seed 固定为0，只把模型 seed 改为1。
- 输入仍为 IC=1、WATER/NITRO=Y、IRRIG/FERTI=L，不修改天气、土壤、品种或初始剖面。

## 执行顺序

1. 检查 `YC2014/seed1_50000steps` 与 `FQ2016/seed1_50000steps` 均不存在。
2. 在 Docker `b2fd6726c8c1`、`/opt/gym_dssat_pdi/bin/python` 中先运行 YC2014 seed1 50K，每5K确定性评估一次。
3. YC 完成并通过输入、预算、动作传输和输出审计后，再串行运行 FQ2016 seed1 50K；禁止并行。
4. 按 total reward 最大、并列取最早 checkpoint 的冻结规则选模型。
5. 与 seed0、null、recorded、DSSAT auto、官方推广 expert 比较；不得因结果不好修改奖励或约束。

## 输出

- 独立 seed1 输入、模型、checkpoint、日值、汇总、PDI 快照和 runtime audit。
- `020_13` 跨 seed 汇总 CSV 与中文实验记录。
- 记录所有异常、修复和结论，不覆盖 020_12 结果。
