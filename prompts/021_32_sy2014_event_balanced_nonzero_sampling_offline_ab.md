# 021_32 SY2014 非零事件等频采样离线 A/B

## 背景

021_31 复现了 021_30 Treatment，并确认非零组内 PER 存在强偏置：action1 在五个事件中本应占 40%，但 updates 251–1000 实际只占 19.425%；两个 action1 均未召回，而 action4/action7 均已召回。

本任务只移除这一项已确认偏置，检验五个非零事件等频出现后，网络能否同时学会五个操作并保持 no-op 准确率。

## A/B

- Control：021_30 Treatment，batch size 32，其中 16 no-op + 16 非零；非零组内按 1-step TD priority 抽样；
- Treatment：同样 16 no-op + 16 非零；五个非零事件每批各出现至少 3 次，额外 1 次按 update 在五个事件间轮换，保证每连续 5 个 batch 完全等频。

Treatment 的 no-op 仍按原组内 PER 抽样。五个非零事件的 priority 继续计算、更新和记录，但不再控制它们之间的采样频率。

## 冻结项

- 完全离线，不调用 DSSAT/PDI，不在线交互；
- 同一示范、full return-to-go、固定观测标准化；
- 同一网络、seed、优化器、学习率、batch size；
- 同一 margin、n-step、L2；
- 两组示范 1-step TD 均不进入总损失，但仍更新 priority；
- no-op 采样机制不变；
- 不调整 reward、动作空间或里程碑；
- 不现场改变事件频率；
- 不自动进入在线 5K。

## 里程碑与门槛

里程碑：`0, 10, 25, 50, 100, 250, 500, 1000`。

单里程碑通过：

- 非零动作召回率 >= 0.8；
- no-op 准确率 >= 0.95；
- 非零动作正 expert-margin 比例 >= 0.8；
- Q 值和参数有限。

整体判定：

- A：Treatment 至少两个连续里程碑通过，Control 不通过——事件级 PER 偏置是关键剩余机制，可另立在线 5K A/B；
- B：Treatment 仅局部通过或召回改善但不能稳定维持联合门槛——等频采样有信号但仍不足；
- C：Treatment 无实质改善——事件级采样偏置不能单独解释剩余失败。

## 强制验证

- Control 共同里程碑必须复现 021_30 Treatment，误差 < 1e-6；
- Treatment 每批严格 16 no-op + 16 非零；
- Treatment 每连续 5 个 batch 中五个事件各累计 16 次；
- 1000 次更新后每个非零事件总计恰好 3200 次；
- 所有更新有限；
- 失败尝试保留，不覆盖旧结果。

## 输出

- `benchmark_results/021_32/021_32_learning_curve.csv`
- `benchmark_results/021_32/021_32_event_predictions.csv`
- `benchmark_results/021_32/021_32_event_sampling.csv`
- `benchmark_results/021_32/021_32_update_diagnostics.csv`
- `benchmark_results/021_32/021_32_validation.json`
- `benchmark_results/021_32/021_32_summary.json`
- PNG/SVG
- `docs/2026-07-15_021_32_sy2014_event_balanced_nonzero_sampling_offline_ab.md`

不自动 Git push。
