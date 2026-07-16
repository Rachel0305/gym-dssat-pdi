# 021_33 SY2014 false-positive 时序邻域离线审计

## 背景

021_32 的事件等频 Treatment 在 500 次更新时达到非零召回率 0.8、正 margin 比例 0.8，但 no-op 准确率为 0.9419，略低于预注册的 0.95；1000 次时 no-op 准确率进一步降到 0.8903。

本任务判断这些 false-positive no-op 是：

1. 集中在 oracle 操作日前后的轻微时序偏移；还是
2. 分散在全生长季的真实过度操作。

## 边界

- 完全离线，不调用 DSSAT/PDI；
- 不在线交互；
- 严格复现 021_32 event-balanced Treatment；
- 不修改 sampler、reward、loss、网络、seed 或里程碑；
- 在 250、500、1000 次更新保存全部 160 个示范状态的 Q 值与预测；
- 不放宽 021_32 的 no-op 门槛；
- 不把“时序接近”直接写成“农艺等价”，后者仍需 DSSAT 前向验证；
- 不自动启动在线 5K。

## 指标

对每个 false-positive no-op 状态记录：

- transition index 与 DAP；
- 预测动作；
- 最近 oracle 事件的 DAP、动作和绝对距离；
- 最近“同预测动作”oracle 事件的绝对距离；
- 是否位于任一 oracle 事件的 ±1、±3、±7 DAP；
- 是否位于同动作 oracle 事件的 ±1、±3、±7 DAP；
- 预测动作 Q-margin。

## 预注册判断

以 500 次更新为主判断：

- A（时序局部）：至少 80% false positive 位于任一 oracle 事件 ±3 DAP，且至少 80% 位于同动作事件 ±3 DAP；
- B（混合）：上述任一比例达到 50%，但未同时达到 80%；
- C（分散过度操作）：两项均低于 50%。

同时报告 250 和 1000 次结果，不用它们替换 500 次主判断。

## 强制验证

- 021_32 Treatment 的共同里程碑最大绝对误差 < 1e-6；
- 500/1000 次 false-positive 数量必须分别与 no-op accuracy 精确对应；
- 所有 Q 值和参数有限；
- 失败尝试保留，不覆盖旧结果。

## 输出

- `benchmark_results/021_33/021_33_all_state_predictions.csv`
- `benchmark_results/021_33/021_33_false_positive_states.csv`
- `benchmark_results/021_33/021_33_temporal_summary.csv`
- `benchmark_results/021_33/021_33_learning_curve.csv`
- `benchmark_results/021_33/021_33_validation.json`
- `benchmark_results/021_33/021_33_summary.json`
- PNG/SVG
- `docs/2026-07-15_021_33_sy2014_false_positive_temporal_neighborhood_audit.md`

不自动 Git push。
