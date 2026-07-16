# 021_31 SY2014 非零示范动作级采样审计

## 边界

本任务完全离线，并通过包装 021_30 已验证的 `stratified_sample()` 捕获真实抽样；未修改 sampler、priority、训练目标、网络或 seed。

## 强制验证

```json
{
  "021_30_treatment_max_abs_errors": {
    "noop_accuracy": 1.1102230246251565e-16,
    "nonzero_action_recall": 0.0,
    "positive_expert_margin_fraction_nonzero": 0.0,
    "mean_expert_margin_nonzero": 8.326672684688674e-17
  },
  "learning_curve_reproduced": true,
  "all_batches_have_16_nonzero": true,
  "all_batches_have_16_noop": true,
  "all_updates_finite": true,
  "passed": true
}
```

## 1000 次更新累计动作采样

| milestone | target_action | event_count | event_count_reference_share | cumulative_sample_count | cumulative_sample_share |
| --- | --- | --- | --- | --- | --- |
| 1000.0 | 1.0 | 2.0 | 0.4 | 2966.0 | 0.185375 |
| 1000.0 | 4.0 | 2.0 | 0.4 | 8068.0 | 0.50425 |
| 1000.0 | 7.0 | 1.0 | 0.2 | 4966.0 | 0.310375 |

## 1000 次更新动作预测

| transition_index | dap | target_action | predicted_action | target_action_recalled | expert_margin |
| --- | --- | --- | --- | --- | --- |
| 42 | 22 | 1 | 0 | False | -0.090569 |
| 49 | 29 | 4 | 4 | True | 0.827499 |
| 62 | 42 | 7 | 7 | True | 0.737976 |
| 76 | 56 | 4 | 4 | True | 0.535753 |
| 99 | 79 | 1 | 0 | False | -4.916401 |

## 预注册判定

- action1 事件数参照份额：`0.40`
- updates 251–1000 的 action1 实际采样份额：`0.194250`
- 两个 action1 均未召回：`True`
- action4/action7 均已召回：`True`
- Branch：`A_strong_within_nonzero_per_bias_support`

## 结论边界

本轮只判断组内 PER 是否对 action1 形成明显低采样。未现场修改采样规则，也未启动在线训练。
