# 012_21 HLA2015 warm-start seed1 Q-value 诊断

## 背景

012_19 warm-start seed0 成功得到 I66/N0；
012_20 warm-start seed1 仍然回到 I48/N150。

二者设置完全一致，唯一变化是 seed。

## 目的

不重新训练，直接加载 012_20 seed1 模型，检查：

1. seed1 在关键状态中是否仍然把高氮动作排在 best-Q；
2. 与 012_19 seed0 的结果相比，seed1 是在哪些 DAP 开始偏向 N150；
3. warm-start 是否被在线训练覆盖。

## 输入模型

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_warmstart_012_19/2015/literature_warmstart_nstep5_ws8_seed1_5000steps/models/dqn_literature_warmstart_probe.zip
```

## 执行要求

- 不训练；
- 使用 Docker 容器 `b2fd6726c8c1`；
- 使用 Python `/opt/gym_dssat_pdi/bin/python`；
- 输出逐日 Q-value CSV、关键 DAP top actions、summary 和中文记录。

## 判断

如果 seed1 在 DAP50–70 左右将 I0_N120 / I0_N160 / I24_N160 等动作排高，则说明：

- warm-start 初始化没有被稳定保留；
- 在线 DQN 更新仍然会把高氮动作价值估高；
- 这条线需要更系统的 imitation/replay，而不是一次性 warm-start。
