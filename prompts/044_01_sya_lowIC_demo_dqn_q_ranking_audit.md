# 044_01 SYA lowIC Demo-DQN Q-ranking audit

## 背景

044_00 证明 Demo-DQN / DQfD 机制链路可以运行，demo replay 和 demo margin loss 都进入了更新日志。但 2K smoke 的 deterministic rollout 仍然是全 no-op。

因此不能直接加训练步数。需要先检查：demo loss 是否真的把 teacher 动作的 Q 值推到 no-op 和其他合法动作之上。

## 本任务目标

不训练新模型，只读取 044_00 的 checkpoint，在 teacher/demo states 上审计 Q 排序：

- teacher 动作是否是合法动作中的 argmax；
- teacher 动作 Q 值是否高于 no-op；
- teacher 动作 Q 值是否高于其他合法动作；
- action1/action2/action3 哪类 teacher 动作最难学。

## 固定边界

- 不训练；
- 不调用 DSSAT 长期 rollout；
- 不改 reward；
- 不改输入数据；
- 只重建 teacher demo buffer，用于获得 observation/mask/action；
- 只读取 044_00 smoke2k 的 1000/2000 checkpoint。

## 分支判据

- A：非零 teacher 动作在 demo states 上多数成为 argmax，说明 DQfD 机制有苗头，可以考虑更长训练；
- B：teacher 动作 Q 排序仍然明显低于 no-op/其他动作，说明当前 demo margin/采样强度不够；
- C：不同动作类别表现分裂，例如 N 学得进去、I 或 I+N 学不进去，需要先处理动作结构或 demo loss。

## 输出

输出逐样本 Q 排序表、按 checkpoint/action/year 汇总表和记录文档。
