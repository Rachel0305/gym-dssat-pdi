# 021_34 SY2014 event-balanced 冻结策略前向评估

## 背景

021_32 的 500-update event-balanced 模型在固定 oracle 状态上达到非零召回率 0.8、no-op 准确率 0.9419。021_33 发现 9 个 false positive 中仅 3 个位于同动作 oracle 事件 ±3 DAP，不能把全部误差解释为轻微日期平移。

固定 oracle 状态分类仍不能等同于自由策略行为，因为自由回放会改变后续状态，同时受共享 7 天间隔、I120/N300 预算和单次上限裁剪。本任务只评估该冻结模型在一个真实 SY2014 生长季中的实际行为。

## 边界

- 重建 021_32 Treatment 到 500 次离线示范更新；
- 严格复现 021_32 的 500-update 指标；
- 模型随后完全冻结，`deterministic=True` 等价贪心动作，无 epsilon 探索；
- 只运行一个 SY2014 DSSAT/PDI 前向生长季；
- 不进行任何在线梯度更新，不写 replay，不继续到 1000 次；
- 不修改 reward、观测、动作、预算、IC、DSSAT 输入或 sampler；
- expert、auto、recorded、null 和 oracle 指标复用 021_18，不重复运行；
- 无论结果如何，不自动启动在线 5K。

## 输出与指标

- 每日 requested action、实际安全灌溉/施氮、reward、Q 值和关键状态；
- 最终 GWAD、CWAD、灌溉量、施氮量、晚期氮、操作次数；
- 与 null、recorded、DSSAT auto、official expert、021_18 oracle 对比；
- PDI 临时目录快照；
- 500-update 模型文件仅本地保存，不纳入 Git。

## 预注册解释

- A：冻结策略产量 >= official expert 11077 kg/ha，且 I<=266、N<=300，同时实际操作时序无明显晚期滥施——可另立在线 5K A/B；
- B：产量达到 recorded 9613 kg/ha、I≤266、N≤300，但未同时满足 A 的 expert 产量与无晚期氮条件——说明离线引导有价值但尚不能作为正式策略；
- C：退化到低产、no-op 或无意义打满预算——离线分类改善未转化为自由策略质量，不进入在线训练。

这些门槛只用于候选分级，不把单次前向结果写成跨 seed 稳定结论。

## 强制验证

- 500-update 离线指标与 021_32 Treatment 最大绝对误差 < 1e-6；
- 前向过程中模型参数哈希前后不变；
- 无 optimizer step、无 replay 写入；
- Q 值和参数有限；
- 生长季正常终止；
- 不覆盖旧结果。

## 输出文件

- `benchmark_results/021_34/021_34_frozen_policy_daily.csv`
- `benchmark_results/021_34/021_34_frozen_policy_summary.json`
- `benchmark_results/021_34/021_34_baseline_comparison.csv`
- `benchmark_results/021_34/021_34_validation.json`
- `benchmark_results/021_34/021_34_summary.json`
- PNG/SVG
- `docs/2026-07-15_021_34_sy2014_event_balanced_frozen_policy_forward_evaluation.md`

不自动 Git push。
