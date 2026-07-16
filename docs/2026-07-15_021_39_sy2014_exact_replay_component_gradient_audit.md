# 021_39 SY2014 精确重放与分项梯度审计记录

## 强制复现

021_35 checkpoint轨迹最大绝对误差：0；逐项复现通过：True。只增加autograd诊断，没有改变optimizer更新。

## 坍缩与恢复期分项梯度

| metric | collapse_median | recovery_median | collapse_to_recovery_ratio | absolute_stage_difference |
| --- | --- | --- | --- | --- |
| grad_td1_agent | 2.50336 | 2.23119 | 1.12198 | 0.27217 |
| grad_tdn_all | 2.37889 | 1.40865 | 1.68877 | 0.97024 |
| grad_margin_demo | 0.54692 | 0.32047 | 1.70662 | 0.22645 |
| cos_td1_tdn | -0.02701 | -0.21858 | -0.12356 | 0.19157 |
| cos_td1_margin | 0.1874 | 0.27194 | 0.68912 | 0.08454 |
| cos_tdn_margin | -0.61794 | -0.40467 | -1.52704 | 0.21327 |

预注册分支：**A**。至少一个分项梯度范数或方向在坍缩/恢复期出现预注册的时段差异。

## 边界

这是同一训练轨迹上的时段对照，仍然是观察性机制证据，不证明某项梯度是根因。本任务不改任何loss权重，也未超过1K。

## 输出

- `benchmark_results/021_39/021_39_component_gradient_update_log.csv`
- `benchmark_results/021_39/021_39_checkpoint_trajectory.csv`
- `benchmark_results/021_39/021_39_collapse_recovery_gradient_comparison.csv`
- `benchmark_results/021_39/021_39_component_gradient_audit.png/.svg`
