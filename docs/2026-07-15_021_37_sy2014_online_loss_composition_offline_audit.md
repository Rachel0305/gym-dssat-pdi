# 021_37 SY2014 在线损失构成离线审计记录

## 边界

只读取021_35的951条更新日志和021_36的checkpoint Q指标；未训练、未调用DSSAT。由于未保存逐样本观测和分项梯度，本结果只能定位聚合loss构成，不能证明因果。

## 坍缩期与恢复期

| component | collapse_median | recovery_median | collapse_to_recovery_ratio | collapse_median_share | recovery_median_share | absolute_share_difference |
| --- | --- | --- | --- | --- | --- | --- |
| td1_agent_only | 38.03601 | 67.63054 | 0.56241 | 0.08035 | 0.19354 | 0.11318 |
| tdn_all | 490.45471 | 287.91837 | 1.70345 | 0.91946 | 0.80636 | 0.1131 |
| margin_demo_only | 0.08968 | 0.03905 | 2.29673 | 0.00017 | 0.00011 | 6e-05 |
| l2 | 0.00072 | 0.00075 | 0.96463 | 0.0 | 0.0 | 0.0 |

- 总梯度中位数collapse/recovery比：1.3749。
- 预注册强候选项：['td1_agent_only', 'tdn_all']。
- 分支：**A**。至少一个损失项在坍缩期与恢复期出现预注册的显著构成变化，可作为下一轮优先候选。

## 结论边界

即使某项loss数值占比很高，也不等同于其参数梯度或决策方向占主导；Huber loss、样本状态和梯度方向均会影响实际更新。本轮不调整任何权重。

## 输出

- `benchmark_results/021_37/021_37_update_log_with_component_shares.csv`
- `benchmark_results/021_37/021_37_interval_summary.csv`
- `benchmark_results/021_37/021_37_collapse_recovery_comparison.csv`
- `benchmark_results/021_37/021_37_online_loss_composition_audit.png/.svg`
