# 021_31 SY2014 非零示范动作级采样审计

## 背景

021_30 将每个 batch 固定为 16 no-op + 16 非零示范后，非零动作召回率从 0 提高到 0.6，同时 no-op 准确率保持约 0.96–0.97。网络稳定学会 DAP29/action4、DAP42/action7、DAP56/action4，却始终漏掉 DAP22/action1 和 DAP79/action1。

Treatment 的非零组内部仍按 1-step TD error 更新 priority。action1 的即时成本为 -15，而 action4/action7 的即时成本绝对值更大，因此存在一个待验证假设：**组内 PER 仍系统性低采样两个 irrigation-only action1 事件。**

## 边界

- 完全离线，不调用 DSSAT/PDI；
- 不在线交互；
- 严格复现 021_30 Treatment 的 16/16 分层采样和 1000 次更新；
- 不修改 sampler、priority、importance weight、reward、loss、网络或 seed；
- 只通过包装现有 `stratified_sample()` 捕获真实抽样结果；
- 不根据结果调整采样比例；
- 不自动启动在线训练。

## 记录内容

对每次更新记录五个非零事件的：

- transition index、DAP、目标动作；
- 更新前 priority；
- 全局 PER 概率；
- 非零组内条件概率；
- 本 batch 实际抽样次数；
- 累计抽样次数。

在 `0, 10, 25, 50, 100, 250, 500, 1000` 里程碑记录五个事件的预测动作、是否召回和 expert Q-margin，并与 021_30 Treatment 严格对齐。

## 预注册参照

五个事件中 action1 有 2 个、action4 有 2 个、action7 有 1 个；若事件等权，其参照份额分别为 0.40、0.40、0.20。

- A（强支持组内 PER 偏置）：250–1000 更新阶段 action1 累计抽样份额低于 0.20，且两个 action1 均未召回，而 action4/7 已召回；
- B（部分支持）：action1 份额为 0.20–0.35，且召回差异与采样差异方向一致；
- C（不支持）：action1 份额 >= 0.35，或 action1 并未显著少采样但仍失败。

阈值只用于本轮机制分级，不用于现场调参。

## 强制验证

- 021_30 Treatment 的共同里程碑指标最大绝对误差 < 1e-6；
- 每个 batch 必须为 16 no-op + 16 非零；
- 五个事件每次更新的 batch 抽样数之和必须为 16；
- 所有更新有限；
- 失败尝试保留，不覆盖旧结果。

## 输出

- `benchmark_results/021_31/021_31_action_sampling_per_update.csv`
- `benchmark_results/021_31/021_31_action_sampling_summary.csv`
- `benchmark_results/021_31/021_31_event_predictions.csv`
- `benchmark_results/021_31/021_31_learning_curve.csv`
- `benchmark_results/021_31/021_31_validation.json`
- `benchmark_results/021_31/021_31_summary.json`
- PNG/SVG
- `docs/2026-07-15_021_31_sy2014_action_level_sampling_audit.md`

不自动 Git push。
