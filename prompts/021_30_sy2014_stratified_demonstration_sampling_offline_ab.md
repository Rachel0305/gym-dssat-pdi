# 021_30 SY2014 分层平衡示范采样离线 A/B

## 背景

021_29 证明，5 个非零示范动作虽然仅占 3.125%，但全局 PER 已将其概率质量提高到中位数 12.33%，因此失败不能简单归因于“完全抽不到”。然而，no-op 样本梯度范数仍是非零动作的约 21.1 倍，两组梯度 cosine 中位数为 -0.429，存在明显的共享网络梯度失衡和冲突。

本任务只检验一个问题：**在不改变 reward、目标、网络和损失权重的情况下，按 no-op/非零动作分层平衡 batch，是否足以让网络同时学会稀疏操作与正确 no-op。**

## A/B 设计

- Control：沿用 021_28 Treatment，即全局 PER、full return-to-go、示范 1-step TD 不进入总损失；
- Treatment：保持冻结 `batch_size=32`，每个 batch 固定 16 个 no-op 与 16 个非零示范动作；两个组内仍依据当前 PER priority 归一化抽样，并计算各组内 importance weight；其他不变。

Treatment 是“类别平衡训练目标”的机制实验，不声称等价于原始全局 PER 的无偏估计。其目的正是检验调整两类样本的有效训练权重能否解决 021_29 已确认的梯度失衡。

## 冻结项

- 完全离线，不调用 DSSAT/PDI，不在线交互；
- 同一 160 条示范、固定标准化、full return-to-go；
- 同一 seed、网络、优化器、学习率、batch size；
- 同一 margin、n-step、L2 权重；
- 两组示范 1-step TD 均不进入总损失；
- 两组 priority 均继续按原始 1-step TD error 更新；
- 不调整 reward、观测、动作空间或训练里程碑；
- 不根据中间结果修改 16/16 比例；
- 无论结果如何，不自动进入在线 5K。

## 里程碑与门槛

里程碑：`0, 10, 25, 50, 100, 250, 500, 1000`。

单里程碑通过：

- 非零动作召回率 >= 0.8；
- no-op 准确率 >= 0.95；
- 非零动作正 expert-margin 比例 >= 0.8；
- Q 值和参数有限。

整体判定：

- A：Treatment 至少两个连续里程碑通过，Control 不通过——类别/梯度平衡是重要机制，可另立在线 A/B；
- B：Treatment 的非零召回或 margin 明显改善，但不能同时维持 no-op 门槛——平衡方向有信号但 16/16 过强或共享表示仍冲突；不现场调比例；
- C：Treatment 无实质改善——分层平衡采样不足以解决，停止继续扫描采样比例。

## 强制验证

- Control 必须复现 021_28 Treatment 的共同里程碑；
- Treatment 每次更新必须严格为 16 no-op + 16 非零动作；
- 两组均不得含 agent 样本；
- 所有 target、Q、梯度和参数必须有限；
- 保存失败尝试，不覆盖既有结果。

## 输出

- `benchmark_results/021_30/021_30_learning_curve.csv`
- `benchmark_results/021_30/021_30_nonzero_event_predictions.csv`
- `benchmark_results/021_30/021_30_update_diagnostics.csv`
- `benchmark_results/021_30/021_30_validation.json`
- `benchmark_results/021_30/021_30_summary.json`
- PNG/SVG
- `docs/2026-07-15_021_30_sy2014_stratified_demonstration_sampling_offline_ab.md`

不自动 Git push。
