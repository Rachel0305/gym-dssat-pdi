# 012_18 HLA2015 文献对齐 + n-step DQN Q-value 诊断

## 背景

012_17 在 012_15 文献式 DQN 基础上只加入 `n_steps=5`：

- 产量基本不变；
- 灌溉从 102 mm 降到 96 mm；
- 施氮仍然是 N150。

因此需要确认：n-step 后，Q 网络是否仍然把含氮动作排高。

## 目的

不重新训练，直接加载 012_17 已训练模型，检查：

1. 每个决策日 25 个动作的 Q-value 排序；
2. 灌溉窗口内 best-Q 动作是否仍然含氮；
3. 施氮窗口内是否仍然偏向高氮动作；
4. 与 012_16 的 n=1 Q-value 诊断相比，n-step 是否真正改善了 Q 排序。

## 输入模型

```text
DSSAT_auto_validation/HLA_2004/hla2015_literature_dqn_nstep_012_17/2015/literature_nstep_nstep5_seed0_5000steps/models/dqn_literature_nstep_probe.zip
```

## 执行要求

- 不训练；
- 使用 Docker 容器 `b2fd6726c8c1`；
- 使用 Python `/opt/gym_dssat_pdi/bin/python`；
- 新建输出目录，不覆盖 012_16；
- 输出逐日 Q-value、关键 DAP top actions、summary 和中文记录。

## 判断

如果含氮动作 best-Q 比例仍然很高，则说明：

- n-step return 单独不能解决氮动作高估；
- 后续应关闭“单独调 n_steps”路线；
- 如果继续算法线，应考虑 prioritized replay 或 warm-start / imitation。
