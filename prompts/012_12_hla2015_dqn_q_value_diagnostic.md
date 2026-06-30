# 012_12 HLA2015 DQN Q值排序诊断 prompt

## 背景

012_11 训练动作覆盖诊断显示：

- 失败 seed0/seed2 在训练过程中并不是没有尝试过灌溉；
- 大多数 episode 都执行过有效灌溉；
- 但最终 deterministic policy 中，seed0 和 seed2 没有保留灌溉动作；
- seed1 保留了 I60/N0 策略，并接近 fixed I60/N0。

因此，下一步需要检查：

> 失败 seed 在最终策略中是否把灌溉动作的 Q 值估低了？

## 本轮目标

加载已经训练好的 HLA2015 DQN 模型，不重新训练，沿 deterministic evaluation 轨迹记录每个 DAP 的 Q 值：

- Q(action 0)：不操作；
- Q(action 1)：灌溉；
- Q(action 2)：施氮；
- Q(action 3)：灌溉+施氮。

重点查看灌溉窗口内：

- seed0/seed2 是否把 action 0 或 action 2 排在 action 1/3 前面；
- seed1 是否在 DAP46/53 附近把 action 1 评为最高；
- seed0 20K 是否比 seed0 5K 的 Q 值排序有所改善但仍不足。

## 模型范围

诊断以下模型：

- seed0 5K；
- seed1 5K；
- seed2 5K；
- seed0 20K。

## 不做的事

- 不训练；
- 不改 reward；
- 不改动作空间；
- 不改模型；
- 不覆盖已有模型；
- 只读取模型并输出 Q 值表和图。

## 预期输出

- 每个模型每个 DAP 的 Q 值日值表；
- 灌溉窗口内最佳 action 分布；
- 关键 DAP 的 Q 值排序表；
- Q 值诊断图；
- 中文实验记录。

