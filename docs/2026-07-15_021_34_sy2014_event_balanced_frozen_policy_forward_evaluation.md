# 021_34 SY2014 event-balanced 冻结策略前向评估记录

## 目的与边界

重建 021_32 Treatment 到 500 次离线更新，随后冻结模型，仅进行一个 SY2014 DSSAT/PDI 生长季的确定性前向评估。无 epsilon、无在线梯度、无 replay 写入；reward、IC、动作空间、预算和 DSSAT 输入均未修改。

## 复现与防污染检查

- 021_32 500-update 指标最大误差：`1.110e-16`；复现通过：`True`。
- 前向前后参数哈希一致：`True`。
- replay agent_size 前/后：`0/0`。
- Q 有限、季节正常终止、资源在 wrapper 预算内：`True/True/True`。
- 在线训练：未启动。

## 冻结策略结果

- 产量：11175.0 kg/ha；生物量：20122.0 kg/ha。
- 灌溉：90.0 mm；施氮：300.0 kg/ha。
- DAP>90 施氮：0.0 kg/ha；非零操作次数：6。
- 预注册分支：**A**。

## 与既有基准比较

| scenario | yield_kg_ha | irrigation_mm | nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg |
| --- | --- | --- | --- | --- | --- |
| null | 5408.0 | 0.0 | 0.0 | 1.22 |  |
| recorded | 9613.0 | 0.0 | 247.0 | 2.16 | 38.9 |
| dssat_auto | 5498.0 | 66.0 | 0.0 | 1.2 |  |
| official_extension_expert | 11077.0 | 266.0 | 300.0 | 2.26 | 36.9 |
| deterministic_oracle_021_18 | 11205.0 | 75.0 | 200.0 | 2.3 | 56.0 |
| event_balanced_500_frozen | 11175.0 | 90.0 | 300.0 | 2.24 | 37.2 |

## 结论

冻结策略达到 official expert 产量，资源不超基准且无 DAP>90 晚期施氮；可另立在线 A/B。

这只是单个冻结模型、单季确定性前向结果，不代表跨 seed 稳定性，也不自动授权在线 5K。

## 输出

- `benchmark_results/021_34/021_34_frozen_policy_daily.csv`
- `benchmark_results/021_34/021_34_frozen_policy_summary.json`
- `benchmark_results/021_34/021_34_baseline_comparison.csv`
- `benchmark_results/021_34/021_34_validation.json`
- `benchmark_results/021_34/021_34_frozen_policy_forward.png/.svg`
- 本地模型 `event_balanced_500_frozen_model.zip`（不纳入 Git）

## 失败尝试

第一次完整运行的数值与最终结果一致，但 Pandas 默认把情景名字符串 `null` 解析为缺失值，导致比较表和图的首个标签为空。该次输出完整保存在 `benchmark_results/021_34_failed_attempt_1_null_label_parsed_as_na/`；修复为 `keep_default_na=False` 后原样重跑，未改变任何实验变量。
