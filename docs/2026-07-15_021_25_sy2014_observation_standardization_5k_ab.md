# 021_25 SY2014 固定观测标准化 5K 严格 A/B 记录

## 设计

Control 和 Treatment 顺序运行；唯一科学变量是是否应用 021_24 冻结的固定观测标准化。两组均使用原始 reward、相同 seed、示范、literature-aligned DQfD 组件和 5K/1K checkpoint 协议。没有训练超过 5K。

## Checkpoint 结果

| arm | checkpoint | yield_kg_ha | biomass_kg_ha | irrigation_mm | nitrogen_kg_ha | late_n_after_dap90_kg_ha | reward_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control_raw_observation | 1000 | 11226.0 | 20313.0 | 120.0 | 300.0 | 100.0 | 4197.852 |
| control_raw_observation | 2000 | 11239.0 | 20441.0 | 105.0 | 300.0 | 200.0 | 4226.493 |
| control_raw_observation | 3000 | 8841.0 | 17394.0 | 15.0 | 100.0 | 0.0 | 2917.812 |
| control_raw_observation | 4000 | 11077.0 | 19783.0 | 45.0 | 150.0 | 50.0 | 4874.279 |
| control_raw_observation | 5000 | 5408.0 | 10603.0 | 0.0 | 0.0 | 0.0 | 0.096 |
| treatment_standardized_observation | 1000 | 5408.0 | 10603.0 | 0.0 | 0.0 | 0.0 | 0.096 |
| treatment_standardized_observation | 2000 | 5408.0 | 10603.0 | 0.0 | 0.0 | 0.0 | 0.096 |
| treatment_standardized_observation | 3000 | 5408.0 | 10603.0 | 0.0 | 0.0 | 0.0 | 0.096 |
| treatment_standardized_observation | 4000 | 5408.0 | 10603.0 | 0.0 | 0.0 | 0.0 | 0.096 |
| treatment_standardized_observation | 5000 | 5408.0 | 10603.0 | 0.0 | 0.0 | 0.0 | 0.096 |

## 训练数值

| arm | median_online_gradient | max_online_gradient | online_gradient_clip_fraction | online_cumulative_parameter_relative_l2 | all_finite |
| --- | --- | --- | --- | --- | --- |
| control_raw_observation | 695.058716 | 6688.615234 | 1.0 | 0.459123 | True |
| treatment_standardized_observation | 0.489394 | 5.604928 | 0.0 | 0.486471 | True |

## 预注册判定

- numeric gate: `True`
- agronomic gate: `False`
- consecutive two passing checkpoints: `False`
- no later collapse: `False`
- branch: `B_numeric_pass_agronomic_fail`

本任务不因结果好坏自动延长到 25K，也不现场修改标准化、reward 或 DQfD 参数。

## 现有 checkpoint 的零训练成本 Q/示范动作审计

Treatment 在 1K–5K 的确定性评估中始终为纯 no-op。为避免立即调参，进一步对既有 checkpoint 在 160 条冻结示范状态上做了离线审计：

- 示范中只有 5/160 个非零动作；始终预测 no-op 的表面准确率本来就有 96.875%。
- Treatment 五个 checkpoint 的整体示范动作准确率均为 96.875%，但非零示范动作召回率均为 0%。
- Treatment 对 5 个非零示范动作的平均 expert-action Q margin 从 -0.687（1K）缓慢改善到 -0.507（5K），始终没有转正；即专家非零动作的 Q 值始终低于其他动作的最大 Q 值。
- 因此 no-op 不是 wrapper 裁剪造成：网络请求动作本身就是 action 0。标准化解决了数值爆炸，却暴露出稀疏示范动作没有被学会的问题。

该审计支持下一步优先做纯离线的“示范预训练充分度学习曲线”，而不是立即延长在线DSSAT训练或现场修改 loss 权重。

## 输出

- `benchmark_results/021_25/training/`
- `benchmark_results/021_25/evaluations/`
- `benchmark_results/021_25/021_25_checkpoint_trajectory.csv`
- `benchmark_results/021_25/021_25_training_audits.csv`
- `benchmark_results/021_25/021_25_preregistered_checks.csv`
- `benchmark_results/021_25/021_25_checkpoint_demo_q_audit.csv`
- `benchmark_results/021_25/021_25_posthoc_q_audit_summary.json`
- `benchmark_results/021_25/021_25_summary.json`
- `benchmark_results/021_25/021_25_observation_standardization_5k_ab.png/.svg`
