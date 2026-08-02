# 043_07 SYA lowIC BC failure mechanism audit

## 背景

043_04 证明 lowIC teacher 轨迹本身有明显年份差异。

043_06 证明：只做 BC imitation 后，policy rollout 仍然几乎是模板化动作，最大 `unique_action_signatures` 只有 2，没有达到预注册的非模板化标准。

因此本任务不再训练新模型，而是诊断 043_06 的失败发生在哪里。

## 核心问题

需要区分两类完全不同的失败：

1. **开环模仿失败**：在 teacher 原始状态上，BC policy 就不能预测 teacher 动作，说明 teacher 标签/样本平衡/动作映射/网络表达有问题。
2. **闭环漂移失败**：在 teacher 原始状态上能预测，但让 policy 自己进入 DSSAT 后，第一批动作改变了后续状态分布，导致后面坍缩成模板。

## 固定边界

- 不训练新 PPO；
- 不调用 PPO `model.learn()`；
- 不改 reward；
- 不改 DSSAT 输入；
- 只读取 043_06 已有 BC dataset、BC-only 模型和 rollout 结果。

## 审计内容

1. 开环预测：
   - 在 `043_06` BC dataset 上，对 epoch 0/5/20/50 的模型计算动作预测；
   - 分别报告总准确率、非零动作准确率、各动作类别准确率、各年份非零动作准确率；
   - 检查模型是否只是在 no-op 和少数动作之间平均化。

2. 闭环 rollout 对比：
   - 读取 043_06 在 2014–2023 的 rollout action sequence；
   - 从 BC dataset 重建 teacher binary action sequence；
   - 对每个年份比较 teacher 序列与 policy 序列是否一致；
   - 检查 policy 序列是否跨年份模板化。

3. 分支判定：
   - 若开环非零动作准确率低于 0.8，判定为 imitation 学习不足；
   - 若开环非零动作准确率较高但闭环仍模板化，判定为闭环状态分布漂移；
   - 若二者都不好，优先处理 imitation 数据/损失/动作映射，不进入 PPO 长训。

## 输出

输出 CSV 和记录文档，为下一步决定是否改 BC 数据构造、改动作空间或改 policy 架构提供依据。
