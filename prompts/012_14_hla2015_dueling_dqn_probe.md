# 012_14 HLA2015 Dueling DQN 最小算法改进验证 prompt

## 背景

012_13 的 Double DQN seed0 5K 没有救回 HLA2015，结果仍然等同 null。

这说明单独修正 DQN target 的 overestimation / action selection 问题不够。

当前任务结构有一个明显特点：

> 大多数 DAP 都应该不操作，只有少数关键窗口需要灌溉。

普通 DQN 直接学习每个 action 的 Q(s,a)。在这种任务里，模型可能很难把“当前状态整体好不好”和“此时某个动作是否比其他动作更好”分开。

## Dueling DQN 是什么

Dueling DQN 把 Q 值拆成两部分：

```text
Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

其中：

- V(s)：当前状态本身的价值；
- A(s,a)：在当前状态下，某个动作相对其他动作的优势。

直观解释：

> 先判断“这个状态整体有多值钱”，再判断“在这个状态下，灌溉是不是比不操作更好”。

这可能更适合本项目这种多数时间 no-op、少数关键动作重要的作物管理任务。

## 本轮目标

实现项目内 `CustomDuelingDQNPolicy`，不修改 site-packages。

先跑 HLA2015：

- seed0；
- 5000 steps；
- economic reward；
- 同样窗口；
- 同样动作空间；
- 同样训练参数；
- 只改 Q 网络结构为 dueling architecture。

## 判定标准

对比：

- DQN seed0 5K：I0/N0，失败；
- DoubleDQN seed0 5K：I0/N0，失败；
- DQN seed1 5K：I60/N0，成功；
- fixed I60/N0：固定反事实近似最优。

如果 Dueling DQN seed0 5K 能学到：

- 有效灌溉；
- 少施氮或不施氮；
- reward 高于 DQN seed0 5K；

说明 dueling architecture 对该任务有帮助。

如果仍然 null，则说明单独改 dueling 也不足。

## 禁止事项

- 不改 reward；
- 不改窗口；
- 不改动作；
- 不加 Double；
- 不加 PER；
- 不加 n-step；
- 不跑多 seed；
- 先只跑 seed0 5K。

