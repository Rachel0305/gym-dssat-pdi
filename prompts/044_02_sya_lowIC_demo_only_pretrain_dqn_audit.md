# 044_02 SYA lowIC demo-only pretrain DQN audit

## 背景

044_00 的轻量 Demo-DQN smoke 中，demo replay 和 margin loss 进入了训练，但 rollout 仍为全 no-op。

044_01 进一步证明：在 teacher/demo states 上，非零 teacher 动作成为 Q argmax 的比例为 0。这说明仅在 online DQN 中混入 demo batch 还不够。

## 本任务目标

先不做 online DQN，只用 teacher demo buffer 预训练 Q 网络，检查 demo-only 是否能把 teacher 正动作的 Q 排序推起来。

如果 demo-only pretrain 都无法使 teacher 动作成为 argmax，则 DQfD 长训没有必要。

如果 demo-only pretrain 成功，再考虑下一步接 online DQN。

## 固定条件

- 输入：lowIC
- 站点：SYA
- observation：043_02 30维天气/预报/归一化 observation
- 动作：binary `[I0/I45] × [N0/N80]`
- teacher：041_02 lowIC teacher，经 043_05 正动作映射
- 不做 PPO
- 不做 online DQN
- 不跑长训练

## 训练内容

只在 demo buffer 上训练：

- demo TD loss
- demo margin loss

快照 epoch 固定：

- 0
- 50
- 200
- 500

## 判据

- 若非零 teacher 动作 argmax rate ≥ 0.8，说明 Q 排序被 demo 成功锚定；
- 若仍接近 0，说明当前 DQfD 损失/动作表达无法解决 no-op 偏好；
- 若不同动作类别表现分裂，记录为动作结构问题。
