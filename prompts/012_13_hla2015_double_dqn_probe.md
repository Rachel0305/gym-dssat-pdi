# 012_13 HLA2015 Double DQN 最小算法改进验证 prompt

## 背景

012_12 的 Q 值诊断显示，HLA2015 DQN 失败 seed 的问题不是没有探索到灌溉，而是最终 Q 网络没有稳定地把关键窗口的灌溉动作估成最高价值。

当前问题可以表述为：

> DQN 在时机敏感的水氮管理任务中，Q 值动作排序不稳定。

因此本轮不再继续调 reward，也不继续简单增加训练步数，而是做一个最小算法改进：Double DQN。

## 为什么先试 Double DQN

Double DQN 的核心思想是：

普通 DQN 用同一个目标网络同时完成：

```text
选出下一步最大 Q 的动作
评估这个动作的 Q 值
```

这容易造成 Q 值过估或动作排序不稳定。

Double DQN 改成：

```text
用当前 q_net 选择下一步动作；
用 target_q_net 评估这个动作的 Q 值。
```

也就是把“选择动作”和“评估动作”拆开。

本项目当前问题正是动作 Q 值排序不稳定，因此 Double DQN 是比继续加 seed 或加步数更有针对性的下一步。

## 本轮目标

实现一个项目内的 `CustomDoubleDQN`，不修改 site-packages。

先在 HLA2015 上跑：

- seed0；
- 5000 steps；
- economic reward；
- 同样的 action space；
- 同样的窗口和预算；
- 同样的 DQN 超参数；
- 只改 TD target 为 Double DQN target。

## 实验设置

- 年份：HLA2015
- 算法：CustomDoubleDQN
- seed：0
- 训练步数：5000
- 奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

- 动作空间：
  - 0：不操作
  - 1：灌溉 30 mm
  - 2：施氮 50 kg/ha
  - 3：灌溉 30 mm + 施氮 50 kg/ha
- 灌溉窗口：DAP 20–35、45–65、70–95
- 施氮窗口：DAP 25–40、55–70

## 对照基准

重点对比：

| 基准 | 表现 |
|---|---|
| DQN seed0 5K | I0/N0，失败 |
| DQN seed1 5K | I60/N0，成功 |
| fixed I60/N0 | 当前 economic reward 下近似最优 |
| DQN seed0 20K | I30/N50，部分改善但仍不理想 |

## 判定标准

如果 Double DQN seed0 5K 学到：

- 接近 I60/N0；
- 或至少学到有效灌溉且不乱施氮；
- reward 明显高于 DQN seed0 5K；

则说明 Double DQN 对 Q 值排序稳定性有帮助。

如果仍然退化为 null 或只施氮不灌溉，则说明 Double DQN 单独不足，需要考虑 Dueling / PER / n-step。

## 禁止事项

- 不改 reward；
- 不改窗口；
- 不改动作空间；
- 不改 site-packages；
- 不同时加入 Dueling/PER/n-step；
- 不直接跑多 seed；
- 先做 seed0 5K。

