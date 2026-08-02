# 044_00 SYA lowIC binary-forecast Demo-DQN / DQfD smoke

## 背景

043_04 证明 lowIC teacher 轨迹有年份差异。043_06/043_07 证明：把 teacher 当逐日标签直接做 PPO policy imitation，仍然容易坍缩成模板。

DQN 是 off-policy，可以反复使用旧经验。因此本任务测试一个更适合 DQN 的 teacher 用法：把 teacher transitions 固定放入 demo replay，每次 DQN 更新时混入 demo batch，并在 demo 样本上加入 margin loss，让 teacher 动作的 Q 值高于其他合法动作。

## 本任务目标

做一个小规模机制 smoke，不追求最终指标，只验证：

1. teacher 轨迹能否合法映射成当前 binary action transitions；
2. demo replay 是否能进入 DQN 更新；
3. demo margin loss 是否被计算并记录；
4. 2K 训练链路是否能保存 checkpoint 并评估。

## 固定条件

- 站点：SYA
- 输入：lowIC 手动修正目录
- observation：沿用 043_02 的 30 维天气/预报/归一化 observation
- 动作：沿用 042_10 binary timing `[I0/I45] × [N0/N80]`
- safety mask：沿用 043_02/042_10
- teacher：沿用 041_02 lowIC teacher，并用 043_05 的正动作映射规则
- 本轮只跑 smoke：默认 2K steps，checkpoint 1000/2000

## Demo-DQN 损失

训练时包含：

- online TD loss：DQN 自己探索样本；
- demo TD loss：teacher demo 样本；
- demo margin loss：在 demo 样本上，teacher 动作 Q 值应高于其他合法动作至少 margin。

这是轻量 DQfD smoke，不包含 PER，不包含大规模超参扫描。

## 停止规则

- demo transition 收集失败，停止；
- demo 样本大量被 mask，停止；
- 2K smoke 不能完成 checkpoint/eval，停止；
- smoke 结果不作为最终算法优劣结论。
