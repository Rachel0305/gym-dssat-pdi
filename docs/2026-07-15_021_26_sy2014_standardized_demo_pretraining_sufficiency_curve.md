# 021_26 SY2014 标准化观测示范预训练充分度学习曲线

## 边界

本任务完全离线，不调用 DSSAT/PDI、不进行在线训练。所有里程碑、DQfD/PER参数、target冻结方式和判据均在运行前固定；结果不能直接用于挑选一个最好预训练步数。

## 学习曲线

| milestone_updates | overall_accuracy | noop_accuracy | nonzero_action_recall | positive_expert_margin_fraction_nonzero | mean_expert_margin_nonzero | median_total_gradient | gradient_clip_fraction | milestone_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.15 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | False |
| 10 | 0.34375 | 0.354839 | 0.0 | 0.0 | -0.186807 | 1.675263 | 0.0 | False |
| 25 | 0.85 | 0.877419 | 0.0 | 0.0 | -0.188598 | 1.434753 | 0.0 | False |
| 50 | 0.95625 | 0.987097 | 0.0 | 0.0 | -0.251126 | 1.199703 | 0.0 | False |
| 100 | 0.96875 | 1.0 | 0.0 | 0.0 | -0.389789 | 1.017283 | 0.0 | False |
| 250 | 0.96875 | 1.0 | 0.0 | 0.0 | -0.65891 | 0.72447 | 0.0 | False |
| 500 | 0.96875 | 1.0 | 0.0 | 0.0 | -0.93199 | 0.574471 | 0.0 | False |
| 1000 | 0.96875 | 1.0 | 0.0 | 0.0 | -3.898948 | 0.569999 | 0.0 | False |
| 2000 | 0.96875 | 1.0 | 0.0 | 0.0 | -50.835308 | 2.119423 | 0.0 | False |
| 5000 | 0.96875 | 1.0 | 0.0 | 0.0 | -206.972946 | 9.611451 | 0.472333 | False |

## 预注册判定

- passing milestones: `[]`
- two consecutive milestones passed: `False`
- branch: `C_more_current_pretraining_not_sufficient`

## 解释边界

该结果只判断“当前固定损失下，单纯延长示范预训练是否足以学习5个稀疏动作”，不代表在线策略、产量或跨seed稳定性。

## 非零示范动作的信用分配审计

五个关键动作位于 DAP22、29、42、56、79，其即时 reward 分别为 `-15, -265, -515, -265, -15`；对应的 5-step return 与即时成本完全相同。唯一正的收获奖励 `5797.424` 位于 transition 159，距这些动作还有 117、110、97、83、60 步，全部远超5步窗口。

因此在当前“target预训练期间冻结 + 5-step return”机制下，五个专家水氮动作收到的是只有资源成本、没有收获收益的TD目标。margin=0.8必须同时对抗绝对值15–515的成本型TD目标；延长同一预训练会不断强化no-op，而不是把收获收益传回早期动作。这解释了非零召回率为何从0到5000次更新始终为0、expert margin反而持续变负。

这项证据把下一步从“继续加预训练次数”收紧为“单独修复示范信用分配”；但本任务没有修改n-step、target同步或reward。

## 输出

- `benchmark_results/021_26/021_26_pretraining_learning_curve.csv`
- `benchmark_results/021_26/021_26_nonzero_event_predictions.csv`
- `benchmark_results/021_26/021_26_update_diagnostics.csv`
- `benchmark_results/021_26/021_26_nonzero_demo_credit_assignment_audit.csv`
- `benchmark_results/021_26/021_26_credit_assignment_summary.json`
- `benchmark_results/021_26/021_26_summary.json`
- `benchmark_results/021_26/021_26_pretraining_sufficiency_curve.png/.svg`
