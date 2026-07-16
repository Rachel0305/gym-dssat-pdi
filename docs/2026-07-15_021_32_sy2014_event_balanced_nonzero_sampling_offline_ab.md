# 021_32 SY2014 非零事件等频采样离线 A/B

## 边界

Control 保留 021_30 的非零组内 PER；Treatment 只把五个非零事件改为每五个 batch 完全等频。no-op 采样、priority 更新、训练目标和所有其他参数未变。全程离线，无 DSSAT/PDI 调用。

## 强制验证

```json
{
  "control_021_30_max_abs_errors": {
    "noop_accuracy": 1.1102230246251565e-16,
    "nonzero_action_recall": 0.0,
    "positive_expert_margin_fraction_nonzero": 0.0,
    "mean_expert_margin_nonzero": 8.326672684688674e-17
  },
  "control_reproduced": true,
  "treatment_all_batches_16_noop": true,
  "treatment_all_batches_16_nonzero": true,
  "each_event_has_16_samples_per_five_update_block": true,
  "each_event_has_3200_total_samples": true,
  "all_updates_finite": true,
  "passed": true
}
```

## 学习曲线

| arm | milestone_updates | noop_accuracy | nonzero_action_recall | positive_expert_margin_fraction_nonzero | mean_expert_margin_nonzero | median_total_gradient | gradient_clip_fraction | milestone_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_within_nonzero_per | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | False |
| control_within_nonzero_per | 10 | 0.322581 | 0.0 | 0.0 | -0.163021 | 1.311631 | 0.0 | False |
| control_within_nonzero_per | 25 | 0.793548 | 0.0 | 0.0 | -0.126586 | 1.203205 | 0.0 | False |
| control_within_nonzero_per | 50 | 0.967742 | 0.0 | 0.0 | -0.11375 | 0.993334 | 0.0 | False |
| control_within_nonzero_per | 100 | 1.0 | 0.2 | 0.2 | -0.091128 | 0.960527 | 0.0 | False |
| control_within_nonzero_per | 250 | 0.980645 | 0.4 | 0.4 | 0.023917 | 1.107218 | 0.0 | False |
| control_within_nonzero_per | 500 | 0.96129 | 0.6 | 0.6 | 0.151063 | 2.148684 | 0.0 | False |
| control_within_nonzero_per | 1000 | 0.967742 | 0.6 | 0.6 | -0.581148 | 10.598829 | 0.538 | False |
| treatment_event_balanced | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | False |
| treatment_event_balanced | 10 | 0.322581 | 0.0 | 0.0 | -0.158035 | 1.933131 | 0.0 | False |
| treatment_event_balanced | 25 | 0.793548 | 0.0 | 0.0 | -0.115212 | 1.769177 | 0.0 | False |
| treatment_event_balanced | 50 | 0.967742 | 0.0 | 0.0 | -0.093656 | 1.356591 | 0.0 | False |
| treatment_event_balanced | 100 | 1.0 | 0.2 | 0.2 | -0.04409 | 1.360205 | 0.0 | False |
| treatment_event_balanced | 250 | 0.980645 | 0.6 | 0.6 | 0.07788 | 1.616913 | 0.0 | False |
| treatment_event_balanced | 500 | 0.941935 | 0.8 | 0.8 | 0.334837 | 2.967169 | 0.0 | False |
| treatment_event_balanced | 1000 | 0.890323 | 0.8 | 0.8 | 0.263342 | 13.846084 | 0.704 | False |

## Treatment 1000 次更新事件结果

| transition_index | dap | target_action | predicted_action | target_action_recalled | expert_margin |
| --- | --- | --- | --- | --- | --- |
| 42 | 22 | 1 | 1 | True | 0.782104 |
| 49 | 29 | 4 | 4 | True | 0.74794 |
| 62 | 42 | 7 | 7 | True | 0.789581 |
| 76 | 56 | 4 | 4 | True | 0.769432 |
| 99 | 79 | 1 | 0 | False | -1.772346 |

## 预注册判定

- Control passing milestones: `[]`
- Treatment passing milestones: `[]`
- Treatment two consecutive pass: `False`
- Branch: `B_event_balance_partial_but_not_stable`

## 联合门槛为何未通过

- 500 次更新时，非零召回率和正 expert-margin 比例均达到 0.8，但 no-op 准确率为 0.9419，低于预注册的 0.95；因此不能判为通过。
- 1000 次更新时，非零召回率仍为 0.8，但 no-op 准确率进一步降到 0.8903，最近区间梯度裁剪比例升至 70.4%，说明继续增加相同离线更新并没有带来稳定收敛。
- 事件等频后，DAP22/action1 已被正确召回；唯一仍未召回的是 DAP79/action1，其 1000 次 expert margin 为 -1.772。
- 当前尚不知道被误判为操作的 no-op 状态，是集中在正确操作日前后的轻微时序偏移，还是分散在整个生长季的真实过度操作。这个区别必须先通过逐状态离线审计确认，不能事后放宽 no-op 门槛，也不能直接进入在线训练。

## 结论边界

`B_event_balance_partial_but_not_stable`。事件等频纠正了 action1 的总体低采样并将召回提高到 0.8，但代价是 no-op 误判随更新增加，且 DAP79 仍未学会。因此当前证据不足以进入在线 5K。

若继续，下一步应只做离线 temporal-neighborhood 审计：定位 500/1000 次时所有 false-positive no-op 的 DAP、预测动作及其与最近 oracle 操作日的距离。该审计只判断“轻微时序偏移”还是“全季过度操作”，不修改训练或判定门槛。
