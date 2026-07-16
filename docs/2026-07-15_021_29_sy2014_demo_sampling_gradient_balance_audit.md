# 021_29 SY2014 示范采样与梯度平衡离线审计

## 边界

本任务完全离线复现 021_28 Treatment，不调用 DSSAT/PDI、不在线训练、不修改任何训练参数。分组损失均按原 batch 总大小归一化，L2 未归入 no-op 或非零样本贡献。

## 复现验证

```json
{
  "reference": "021_28 treatment_mask_demo_td1",
  "reference_milestones_all_matched": true,
  "additional_audit_milestones": [
    1
  ],
  "max_abs_errors": {
    "noop_accuracy": 5.551115123125783e-17,
    "nonzero_action_recall": 0.0,
    "positive_expert_margin_fraction_nonzero": 0.0,
    "mean_expert_margin_nonzero": 8.326672684688674e-17
  },
  "all_finite": true,
  "max_data_loss_reconstruction_error": 0.00048828125,
  "max_data_loss_reconstruction_relative_error": 2.1645417540645494e-07,
  "passed": true
}
```

## PER 采样与分组梯度

| update | sample_nonzero_count | sample_nonzero_fraction | nonzero_probability_mass | terminal_probability_mass | noop_gradient_norm | nonzero_gradient_norm | noop_to_nonzero_gradient_norm_ratio | noop_nonzero_gradient_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1.0 | 0.03125 | 0.03125 | 0.00625 | 3.249017 | 0.151776 | 21.406689 | 0.047987 |
| 10.0 | 6.0 | 0.1875 | 0.123 | 0.148526 | 2.483123 | 0.153452 | 16.18173 | 0.043901 |
| 25.0 | 2.0 | 0.0625 | 0.130339 | 0.14632 | 2.899214 | 0.070557 | 41.090439 | -0.220225 |
| 50.0 | 3.0 | 0.09375 | 0.158053 | 0.137274 | 2.782098 | 0.112627 | 24.70192 | -0.387823 |
| 100.0 | 9.0 | 0.28125 | 0.149126 | 0.129516 | 2.916899 | 0.316868 | 9.205397 | -0.470999 |
| 250.0 | 3.0 | 0.09375 | 0.123539 | 0.107195 | 3.678445 | 0.177448 | 20.729668 | -0.480235 |
| 500.0 | 2.0 | 0.0625 | 0.078257 | 0.067079 | 11.468885 | 0.467498 | 24.532484 | -0.540364 |
| 1000.0 | 2.0 | 0.0625 | 0.046146 | 0.037048 | 41.05328 | 2.346927 | 17.492353 | -0.540933 |

## 学习结果复现

| milestone_updates | noop_accuracy | nonzero_action_recall | positive_expert_margin_fraction_nonzero | mean_expert_margin_nonzero |
| --- | --- | --- | --- | --- |
| 0.0 | 0.154839 | 0.0 | 0.0 | -0.197212 |
| 1.0 | 0.180645 | 0.0 | 0.0 | -0.196009 |
| 10.0 | 0.354839 | 0.0 | 0.0 | -0.183397 |
| 25.0 | 0.864516 | 0.0 | 0.0 | -0.180022 |
| 50.0 | 1.0 | 0.0 | 0.0 | -0.26443 |
| 100.0 | 1.0 | 0.0 | 0.0 | -0.504978 |
| 250.0 | 1.0 | 0.0 | 0.0 | -1.424533 |
| 500.0 | 1.0 | 0.0 | 0.0 | -5.83414 |
| 1000.0 | 1.0 | 0.0 | 0.0 | -27.375565 |

## 描述性结果

- 原始非零动作占比：`0.0312`
- 非零动作 PER 概率质量中位数：`0.1233`
- batch 非零动作占比中位数：`0.0781`
- no-op / 非零梯度范数比中位数：`21.0682`
- 两组梯度 cosine 中位数：`-0.4294`
- 预注册解释分支：`B_nonzero_sampled_but_gradients_conflict`

## 结论边界

本任务只描述 021_28 Treatment 中 PER 采样和两类示范梯度的实际结构，不现场修改采样、损失权重或网络，也不自动启动在线训练。
