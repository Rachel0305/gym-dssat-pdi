# 021_24 SY2014 固定观测标准化短诊断记录

## 边界与防止乱调

本任务在运行前固定了标准化公式、统计来源、零方差处理、100+100 步规模和三分支判定。恢复 021_22 原始 reward，唯一科学变量是用冻结示范集统计量对同一 25 维信息做固定标准化。没有添加特征、改变 IC/reward/action/loss，也没有进入 5K。

## 实现校验

- pre-run checks: `True`
- post-run checks: `True`
- 非常量维标准化均值最大绝对误差：`1.83e-07`
- 非常量维标准差偏离 1 的最大误差：`4.59e-08`
- 逆变换最大误差：`0.000838`
- 在线 reward 改动误差：`0`

## 早期生长状态是否仍可区分

| observation_label | n_states | minimum | maximum | mean | std | unique_count |
| --- | --- | --- | --- | --- | --- | --- |
| grnwt | 70 | -0.484523 | -0.484523 | -0.484523 | 0.0 | 1 |
| topwt | 70 | -0.795586 | -0.71169 | -0.785562 | 0.019368 | 26 |

`unique_count` 和组内标准差用于确认固定标准化没有把早期 biomass 状态压成同一个常数；这不等同于证明网络一定能够学会利用差异。

早期 `grnwt` 只有一个取值，是因为原始示范数据在 DAP<50 时尚未形成籽粒，原始 `grnwt` 本来就是常数 0，并非标准化造成的信息丢失。早期 `topwt` 仍保留 26 个不同取值。

## 与 021_22 原始观测基准对比

| phase | metric | raw_observation_021_22 | standardized_observation_021_24 | standardized_to_raw_ratio |
| --- | --- | --- | --- | --- |
| pretrain | gradient_clip_fraction | 1.0 | 0.0 | 0.0 |
| pretrain | median_total_gradient_norm | 1274.074646 | 1.135806 | 0.000891 |
| pretrain | cumulative_parameter_relative_l2 | 0.045374 | 0.077181 | 1.700997 |
| pretrain | median_td_1_weighted | 64.897156 | 109.626854 | 1.68924 |
| pretrain | median_td_n_weighted | 174.804863 | 241.564407 | 1.381909 |
| online | gradient_clip_fraction | 1.0 | 0.0 | 0.0 |
| online | median_total_gradient_norm | 387.18338 | 0.995583 | 0.002571 |
| online | cumulative_parameter_relative_l2 | 0.0356 | 0.036507 | 1.025473 |
| online | median_td_1_weighted | 56.066235 | 73.539017 | 1.311645 |
| online | median_td_n_weighted | 124.053558 | 112.818932 | 0.909437 |

## 预注册结论

`A_observation_scale_participant`

该结论只针对 100+100 步的梯度动力学，不代表长期策略已经改善，也不自动授权 5K。

需要同时保留两个反例：标准化后的 TD loss 没有下降；预训练累计参数相对 L2 变化从 0.0454 增到 0.0772（约增加 70%），在线累计变化则基本持平。因此目前只能确认“原始输入尺度造成了极大的瞬时梯度并迫使每次更新裁剪”，不能声称长期振荡、施氮时序或策略坍缩已经解决。

## 输出

- `benchmark_results/021_24/021_24_observation_scaler.csv`
- `benchmark_results/021_24/021_24_growth_stage_distribution.csv`
- `benchmark_results/021_24/021_24_pretrain_update_log.csv`
- `benchmark_results/021_24/021_24_online_update_log.csv`
- `benchmark_results/021_24/021_24_agent_interactions.csv`
- `benchmark_results/021_24/021_24_standardized_stage_summary.csv`
- `benchmark_results/021_24/021_24_stage_comparison.csv`
- `benchmark_results/021_24/021_24_validation.json`
- `benchmark_results/021_24/021_24_summary.json`
- `benchmark_results/021_24/021_24_observation_standardization_diagnostic.png/.svg`
