# 021_30 SY2014 分层平衡示范采样离线 A/B

## 边界

完全离线。Control 使用全局 PER；Treatment 保持冻结 batch size 32，每个 batch 固定 16 no-op + 16 非零示范，并在组内按 priority 抽样。其他训练目标和参数完全相同，两组均屏蔽示范 1-step TD 梯度。

## 强制验证

```json
{
  "control_021_28_max_abs_errors": {
    "noop_accuracy": 5.551115123125783e-17,
    "nonzero_action_recall": 0.0,
    "positive_expert_margin_fraction_nonzero": 0.0,
    "mean_expert_margin_nonzero": 8.326672684688674e-17
  },
  "control_reproduced": true,
  "frozen_batch_size_is_32": true,
  "treatment_all_batches_16_noop": true,
  "treatment_all_batches_16_nonzero": true,
  "all_samples_demonstrations": true,
  "all_updates_finite": true,
  "passed": true
}
```

## 学习曲线

| arm | milestone_updates | noop_accuracy | nonzero_action_recall | positive_expert_margin_fraction_nonzero | mean_expert_margin_nonzero | median_total_gradient | gradient_clip_fraction | milestone_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_global_per | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | False |
| control_global_per | 10 | 0.354839 | 0.0 | 0.0 | -0.183397 | 2.683008 | 0.0 | False |
| control_global_per | 25 | 0.864516 | 0.0 | 0.0 | -0.180022 | 2.558889 | 0.0 | False |
| control_global_per | 50 | 1.0 | 0.0 | 0.0 | -0.26443 | 2.614524 | 0.0 | False |
| control_global_per | 100 | 1.0 | 0.0 | 0.0 | -0.504978 | 2.864073 | 0.0 | False |
| control_global_per | 250 | 1.0 | 0.0 | 0.0 | -1.424533 | 2.644943 | 0.0 | False |
| control_global_per | 500 | 1.0 | 0.0 | 0.0 | -5.83414 | 5.828149 | 0.096 | False |
| control_global_per | 1000 | 1.0 | 0.0 | 0.0 | -27.375565 | 23.678511 | 0.996 | False |
| treatment_stratified_16_16 | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | False |
| treatment_stratified_16_16 | 10 | 0.322581 | 0.0 | 0.0 | -0.163021 | 1.311631 | 0.0 | False |
| treatment_stratified_16_16 | 25 | 0.793548 | 0.0 | 0.0 | -0.126586 | 1.203205 | 0.0 | False |
| treatment_stratified_16_16 | 50 | 0.967742 | 0.0 | 0.0 | -0.11375 | 0.993334 | 0.0 | False |
| treatment_stratified_16_16 | 100 | 1.0 | 0.2 | 0.2 | -0.091128 | 0.960527 | 0.0 | False |
| treatment_stratified_16_16 | 250 | 0.980645 | 0.4 | 0.4 | 0.023917 | 1.107218 | 0.0 | False |
| treatment_stratified_16_16 | 500 | 0.96129 | 0.6 | 0.6 | 0.151063 | 2.148684 | 0.0 | False |
| treatment_stratified_16_16 | 1000 | 0.967742 | 0.6 | 0.6 | -0.581148 | 10.598829 | 0.538 | False |

## 预注册判定

- Control passing milestones: `[]`
- Treatment passing milestones: `[]`
- Treatment two consecutive pass: `False`
- Branch: `B_partial_signal_but_joint_gate_not_stable`

## 动作级核对

- 500 和 1000 次更新时，no-op 准确率分别为 96.1% 和 96.8%，说明非零召回改善并不是靠大面积误触发操作换来的。
- Treatment 在 500 和 1000 次更新均正确召回 DAP29/action4、DAP42/action7、DAP56/action4 三个含氮操作。
- DAP22/action1 和 DAP79/action1 两个仅灌溉 15 mm 的操作始终未被正确召回；其中 DAP79 的 expert margin 在 1000 次更新时降至 -4.916，导致五个事件的平均 margin 再次转负。
- 一个需要另行验证的假设是：Treatment 虽然把“非零组”整体提升到 50%，但组内仍按 1-step TD error 更新和抽样；含氮动作即时成本绝对值较大，而 action1 即时成本仅为 -15，因此组内 PER 仍可能偏向含氮动作。当前结果尚未直接记录每个动作的累计采样次数，不能把该假设写成已确认机制。

## 结论边界

`B_partial_signal_but_joint_gate_not_stable`。分层平衡已产生真实但不完整的改善：非零召回从 0 提高到 0.6，同时保持约 0.96–0.97 的 no-op 准确率；但尚未达到 0.8 的预注册门槛，也未学会两个 irrigation-only 示范动作，后期 margin 仍出现恶化。因此当前结果不足以进入在线 5K。

Treatment 是类别平衡机制实验，不声称是原始全局 PER 目标的无偏估计。本轮没有调用 DSSAT、没有在线训练，也没有现场修改 16/16 比例或其他超参数。下一步若继续，应先做动作级采样审计，确认组内 PER 是否系统性低采样 action1，而不是直接调整比例。
