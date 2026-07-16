# 021_38 SY2014 demonstration梯度冲突离线审计记录

## 边界

只在五个已有checkpoint和同一组160条oracle demonstration上计算梯度；无optimizer.step、无训练、无DSSAT。采用no-op/nonzero各0.5的group-balanced期望权重。agent梯度未保存，不能在本轮恢复。

## 结果

| checkpoint | demo_nstep_loss_group_balanced | demo_margin_loss_group_balanced | grad_norm_demo_nstep | grad_norm_demo_margin | nstep_to_margin_grad_norm_ratio | cosine_demo_nstep_vs_margin | cosine_nstep_noop_vs_nonzero | conflict_checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2171.26929 | 0.29796 | 6.52941 | 3.33307 | 1.95898 | -0.40085 | 0.09057 | False |
| 250 | 2167.5166 | 1.10147 | 9.46038 | 7.07363 | 1.33741 | -0.49367 | 0.08813 | False |
| 500 | 2166.78613 | 1.56463 | 10.34959 | 7.84052 | 1.32001 | -0.5115 | 0.0715 | False |
| 750 | 2167.62451 | 0.88712 | 10.02593 | 7.36062 | 1.3621 | -0.49667 | 0.06577 | False |
| 1000 | 2168.604 | 0.36757 | 9.61903 | 3.29581 | 2.91856 | -0.24896 | 0.06172 | False |

预注册分支：**B**。demo n-step/margin梯度冲突指标在时间上没有区分力，不能解释坍缩与恢复。

target network在五个checkpoint中的唯一哈希数：1，符合1K内target冻结设置。

## 限制

这是固定demonstration上的局部梯度诊断，不等同于真实混合batch的总更新方向，也不构成因果证明。本轮不调整margin、n-step或采样权重。

## 输出

- `benchmark_results/021_38/021_38_demo_gradient_conflict_by_checkpoint.csv`
- `benchmark_results/021_38/021_38_validation.json`
- `benchmark_results/021_38/021_38_summary.json`
- `benchmark_results/021_38/021_38_demo_gradient_conflict.png/.svg`
