# 021_28 SY2014 示范 1-step TD 冲突屏蔽离线 A/B

## 目的与边界

021_27 证明五个非零示范动作上，负的即时成本 1-step TD 与正的完整季节 return TD 形成方向相反的饱和梯度。本任务只检验：在示范样本上屏蔽冲突的 1-step TD 梯度，是否能让网络学会这些稀疏非零动作。

本任务完全离线，不调用 DSSAT/PDI，不进行在线环境交互。两组均使用相同的 160 条示范、固定标准化、完整 return-to-go、网络初始化、PER、margin、L2 和 priority 更新规则。唯一差异是示范 1-step TD 是否进入总损失。

## 实现一致性验证

```json
{
  "same_first_sample_indices": true,
  "same_first_target_1": true,
  "same_first_target_n": true,
  "control_td1_weight": 1.0,
  "treatment_td1_weight": 0.0,
  "all_samples_demonstrations": true,
  "all_updates_finite": true,
  "no_dssat_or_online_training": true,
  "passed": true
}
```

## 学习曲线

| arm | milestone_updates | noop_accuracy | nonzero_action_recall | positive_expert_margin_fraction_nonzero | mean_expert_margin_nonzero | median_td_1_weighted | median_td_n_weighted | median_margin_weighted | median_total_gradient | gradient_clip_fraction | milestone_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_keep_td1 | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | nan | nan | nan | False |
| control_keep_td1 | 10 | 0.354839 | 0.0 | 0.0 | -0.18463 | 117.265873 | 2278.358521 | 0.674767 | 2.780484 | 0.0 | False |
| control_keep_td1 | 25 | 0.864516 | 0.0 | 0.0 | -0.183434 | 120.600273 | 2006.002197 | 0.599938 | 2.534765 | 0.0 | False |
| control_keep_td1 | 50 | 1.0 | 0.0 | 0.0 | -0.269113 | 111.584503 | 1879.646362 | 0.529231 | 2.435537 | 0.0 | False |
| control_keep_td1 | 100 | 1.0 | 0.0 | 0.0 | -0.493644 | 104.435349 | 2036.938721 | 0.384397 | 2.468065 | 0.0 | False |
| control_keep_td1 | 250 | 1.0 | 0.0 | 0.0 | -0.897725 | 99.718277 | 1975.388733 | 0.106666 | 0.830138 | 0.0 | False |
| control_keep_td1 | 500 | 1.0 | 0.0 | 0.0 | -0.785405 | 104.027401 | 2059.112183 | 0.072581 | 0.343395 | 0.0 | False |
| control_keep_td1 | 1000 | 1.0 | 0.0 | 0.0 | -1.013211 | 93.882069 | 1992.289551 | 0.055498 | 0.752318 | 0.0 | False |
| treatment_mask_demo_td1 | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | nan | nan | nan | False |
| treatment_mask_demo_td1 | 10 | 0.354839 | 0.0 | 0.0 | -0.183397 | 117.265923 | 2278.359863 | 0.674765 | 2.683008 | 0.0 | False |
| treatment_mask_demo_td1 | 25 | 0.864516 | 0.0 | 0.0 | -0.180022 | 120.600388 | 2006.029785 | 0.600015 | 2.558889 | 0.0 | False |
| treatment_mask_demo_td1 | 50 | 1.0 | 0.0 | 0.0 | -0.26443 | 111.590904 | 1878.016113 | 0.526535 | 2.614524 | 0.0 | False |
| treatment_mask_demo_td1 | 100 | 1.0 | 0.0 | 0.0 | -0.504978 | 103.317116 | 2040.010071 | 0.37853 | 2.864073 | 0.0 | False |
| treatment_mask_demo_td1 | 250 | 1.0 | 0.0 | 0.0 | -1.424533 | 89.766743 | 1938.471863 | 0.103833 | 2.644943 | 0.0 | False |
| treatment_mask_demo_td1 | 500 | 1.0 | 0.0 | 0.0 | -5.83414 | 93.00526 | 1944.953369 | 0.120349 | 5.828149 | 0.096 | False |
| treatment_mask_demo_td1 | 1000 | 1.0 | 0.0 | 0.0 | -27.375565 | 85.889824 | 1890.986267 | 0.36205 | 23.678511 | 0.996 | False |

## 预注册判定

- Control passing milestones: `[]`
- Treatment passing milestones: `[]`
- Treatment consecutive pass: `False`
- Branch: `C_masking_demo_td1_not_sufficient`

## 结论

`C_masking_demo_td1_not_sufficient`。Treatment 在全部里程碑的非零动作召回率均为 0；到 1000 次更新时仍把全部状态预测为 no-op，非零示范动作平均 Q-margin 从初始的 -0.197 恶化到 -27.376，最近区间梯度裁剪比例达到 99.6%。因此，屏蔽示范 1-step TD 不仅没有解决稀疏操作学习问题，也没有产生可进入在线训练的局部正信号。

该结果说明 021_27 识别出的 1-step/full-return 梯度抵消是真实冲突，但不是唯一机制。下一步应先离线拆分 PER 采样构成，以及 no-op 与非零示范样本各自的梯度贡献和方向；不应直接继续增加训练步数或调整损失权重。

本轮只回答离线示范信用分配问题；未自动启动在线 5K 训练，也未修改 reward、观测、PER、网络或动作空间。

## 输出

- `benchmark_results/021_28/021_28_learning_curve.csv`
- `benchmark_results/021_28/021_28_nonzero_event_predictions.csv`
- `benchmark_results/021_28/021_28_update_diagnostics.csv`
- `benchmark_results/021_28/021_28_implementation_validation.json`
- `benchmark_results/021_28/021_28_summary.json`
- `benchmark_results/021_28/021_28_demo_td1_conflict_mask_ab.png/.svg`
