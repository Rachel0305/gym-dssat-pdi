# 021_33 SY2014 false-positive 时序邻域离线审计

## 边界

本任务严格复现 021_32 event-balanced Treatment，只在 250/500/1000 次更新读取全部状态预测；未修改训练，也未调用 DSSAT/PDI。“时序接近”不等于“农艺等价”。

## 强制验证

```json
{
  "021_32_treatment_max_abs_errors": {
    "noop_accuracy": 1.1102230246251565e-16,
    "nonzero_action_recall": 0.0,
    "positive_expert_margin_fraction_nonzero": 0.0,
    "mean_expert_margin_nonzero": 8.326672684688674e-17
  },
  "learning_curve_reproduced": true,
  "false_positive_count_checks": {
    "500": {
      "expected": 9,
      "observed": 9,
      "match": true
    },
    "1000": {
      "expected": 17,
      "observed": 17,
      "match": true
    }
  },
  "false_positive_counts_match": true,
  "all_q_values_finite": true,
  "all_updates_finite": true,
  "passed": true
}
```

## 时序邻域汇总

| milestone_updates | false_positive_count | noop_state_count | noop_accuracy_from_states | fraction_within_any_1dap | fraction_within_same_action_1dap | fraction_within_any_3dap | fraction_within_same_action_3dap | fraction_within_any_7dap | fraction_within_same_action_7dap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250.0 | 3.0 | 155.0 | 0.980645 | 0.666667 | 0.666667 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| 500.0 | 9.0 | 155.0 | 0.941935 | 0.444444 | 0.333333 | 0.666667 | 0.333333 | 0.777778 | 0.444444 |
| 1000.0 | 17.0 | 155.0 | 0.890323 | 0.235294 | 0.176471 | 0.470588 | 0.294118 | 0.941176 | 0.588235 |

## 500 次更新 false positives

| transition_index | dap | predicted_action | nearest_oracle_dap | nearest_oracle_action | nearest_oracle_distance_dap | nearest_same_action_oracle_dap | nearest_same_action_distance_dap | predicted_q_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32.0 | 12.0 | 1.0 | 22.0 | 1.0 | 10.0 | 22.0 | 10.0 | 0.046163 |
| 41.0 | 21.0 | 1.0 | 22.0 | 1.0 | 1.0 | 22.0 | 1.0 | 0.598069 |
| 61.0 | 41.0 | 7.0 | 42.0 | 7.0 | 1.0 | 42.0 | 1.0 | 0.304554 |
| 68.0 | 48.0 | 7.0 | 42.0 | 7.0 | 6.0 | 42.0 | 6.0 | 0.040481 |
| 75.0 | 55.0 | 4.0 | 56.0 | 4.0 | 1.0 | 56.0 | 1.0 | 0.002409 |
| 77.0 | 57.0 | 7.0 | 56.0 | 4.0 | 1.0 | 42.0 | 15.0 | 0.094316 |
| 78.0 | 58.0 | 1.0 | 56.0 | 4.0 | 2.0 | 79.0 | 21.0 | 0.09225 |
| 79.0 | 59.0 | 1.0 | 56.0 | 4.0 | 3.0 | 79.0 | 20.0 | 0.086065 |
| 87.0 | 67.0 | 1.0 | 56.0 | 4.0 | 11.0 | 79.0 | 12.0 | 0.008428 |

## 预注册判定

- 500 次 false-positive 数：`9`
- 任一 oracle 事件 ±3 DAP 比例：`0.666667`
- 同动作 oracle 事件 ±3 DAP 比例：`0.333333`
- Branch：`B_mixed_temporal_and_dispersed_false_positives`

## 结论边界

`B_mixed_temporal_and_dispersed_false_positives`。该结论只区分 false positive 的时序分布，不证明相邻日期操作具有相同农艺效果，也未自动启动在线训练。
