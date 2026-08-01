# 042_04 SYA lowIC 042_02 天气/胁迫响应性审计记录

## 任务性质

- 只读已有 `042_02_rerun100k` daily/evaluation 结果。
- 不训练、不运行 DSSAT、不修改 checkpoint。
- 目的：判断较好 checkpoint，尤其 75K，是否真的根据天气/胁迫做出年份差异化响应。

## 数据限制

- `041_03` 评估 daily CSV 没有直接保存 tmin/future7 weather observation，因此本任务用站点天气表和 planting date 重建 today/past7/future7 天气特征。
- 部分 daily 行 DAP 早期记录为 0；本任务把 DAP0 视为播种日起点用于日期重建。
- 本审计只能说明相关性和模板化程度，不是动作必要性的因果反事实。

## 动作多样性汇总

| checkpoint_step | years | unique_action_signatures | mean_total_irrigation | sd_total_irrigation | mean_total_nitrogen | sd_total_nitrogen | mean_irrigation_events | sd_irrigation_events | mean_nitrogen_events | sd_nitrogen_events | pairwise_hamming_mean | pairwise_hamming_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 1 | 240.0 | 0.0 | 240.0 | 0.0 | 6.0 | 0.0 | 3.0 | 0.0 | 5.4444 | 15.0 |
| 25000 | 10 | 1 | 225.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 5.4444 | 15.0 |
| 50000 | 10 | 1 | 225.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 5.4444 | 15.0 |
| 75000 | 10 | 1 | 225.0 | 0.0 | 240.0 | 0.0 | 5.0 | 0.0 | 3.0 | 0.0 | 5.4444 | 15.0 |
| 100000 | 10 | 1 | 225.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 5.4444 | 15.0 |

## 性能 + 多样性合并表

| checkpoint_step | years | unique_action_signatures | mean_total_irrigation | sd_total_irrigation | mean_total_nitrogen | sd_total_nitrogen | mean_irrigation_events | sd_irrigation_events | mean_nitrogen_events | sd_nitrogen_events | pairwise_hamming_mean | pairwise_hamming_max | mean_yield | mean_wp_et | mean_pfp_n | any_metric_win_years | all3_win_years | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 1 | 240.0 | 0.0 | 240.0 | 0.0 | 6.0 | 0.0 | 3.0 | 0.0 | 5.4444 | 15.0 | 9955.0 | 2.104 | 41.49 | 10 | 2 | 0.8064 | 0.2593 |
| 25000 | 10 | 1 | 225.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 5.4444 | 15.0 | 3847.4 | 0.893 |  | 0 | 0 | 0.0 | 0.556 |
| 50000 | 10 | 1 | 225.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 5.4444 | 15.0 | 3847.4 | 0.893 |  | 0 | 0 | 0.0 | 0.556 |
| 75000 | 10 | 1 | 225.0 | 0.0 | 240.0 | 0.0 | 5.0 | 0.0 | 3.0 | 0.0 | 5.4444 | 15.0 | 9931.7 | 2.145 | 41.36 | 10 | 6 | 0.8592 | 0.238 |
| 100000 | 10 | 1 | 225.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 5.4444 | 15.0 | 3847.4 | 0.893 |  | 0 | 0 | 0.0 | 0.556 |

## 75K 最强关联特征

| checkpoint_step | action_type | feature | n_action_days | mean_on_action_days | mean_on_no_action_days | standardized_difference |
| --- | --- | --- | --- | --- | --- | --- |
| 75000 | irrigation | topwt | 50 | 1805.6452 | 5091.0218 | -0.5608 |
| 75000 | irrigation | dap | 50 | 45.0 | 66.4418 | -0.5395 |
| 75000 | irrigation | grnwt | 50 | 19.3448 | 1672.1484 | -0.5364 |
| 75000 | irrigation | tmin_today_reconstructed | 50 | 14.792 | 16.516 | -0.3133 |
| 75000 | irrigation | future7_tmean_reconstructed | 50 | 21.1481 | 22.1496 | -0.2492 |
| 75000 | nitrogen | future7_tmean_reconstructed | 30 | 15.3374 | 22.2639 | -1.779 |
| 75000 | nitrogen | tmin_today_reconstructed | 30 | 7.47 | 16.6531 | -1.7176 |
| 75000 | nitrogen | dap | 30 | 8.3333 | 66.9386 | -1.502 |
| 75000 | nitrogen | topwt | 30 | 7.04 | 5082.325 | -0.8685 |
| 75000 | nitrogen | past7_rain_reconstructed | 30 | 7.6567 | 25.0138 | -0.5479 |

## 判定

- 分支：`A_75k_nonzero_management_is_template_like`
- 75K unique action signatures：1
- 75K pairwise Hamming mean：5.4444
- 说明：pairwise Hamming 包含所有日步和后期 no-op 序列，受不同年份季节长度影响；本任务判定以非零措施 signature 为主。
- 如果分支为 A，说明 75K 的好指标更接近固定管理模板带来的结果，而不是明确的天气/胁迫响应策略。
- 下一步若继续优化，应把“动作多样性/天气响应性”写入 checkpoint selection guardrail，而不是只按 endpoint 指标选点。

## 输出

- action sequence: `benchmark_results\042_04_sya_lowIC_04202_weather_responsiveness_audit\tables\042_04_action_sequence_by_year_checkpoint.csv`
- diversity: `benchmark_results\042_04_sya_lowIC_04202_weather_responsiveness_audit\tables\042_04_action_diversity_by_checkpoint.csv`
- association: `benchmark_results\042_04_sya_lowIC_04202_weather_responsiveness_audit\tables\042_04_weather_stress_action_association.csv`
- top associations: `benchmark_results\042_04_sya_lowIC_04202_weather_responsiveness_audit\tables\042_04_top_action_association_features.csv`
- performance + diversity: `benchmark_results\042_04_sya_lowIC_04202_weather_responsiveness_audit\tables\042_04_performance_plus_diversity.csv`
