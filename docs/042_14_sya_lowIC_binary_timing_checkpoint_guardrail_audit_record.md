# 042_14 SYA lowIC binary-timing PPO checkpoint 综合 guardrail 审计记录

## 结论先说

- 通过草案 guardrail 的 checkpoint：25000；按产量比例优先的候选为 25000。
- 本任务不训练、不调参，只读取 042_11 已有 checkpoint 结果。
- 这里的 guardrail 是诊断性草案，用于把“指标表现”和“动作合理性”同时放进 checkpoint 审计；不是最终冻结协议。

## Guardrail 草案

| min_unique_action_signatures | min_mean_yield_ratio_vs_official_expert | max_low_yield_years_below_90pct_expert | max_severe_swfac_failure_years | max_fixed_event_slots |
| --- | --- | --- | --- | --- |
| 5 | 0.9 | 2 | 2 | 3 |

## Checkpoint 综合表

| checkpoint_step | validation_years | mean_yield | mean_official_expert_yield | mean_yield_ratio_vs_official_expert | low_yield_years_below_90pct_expert | mean_total_irrigation | mean_total_n | mean_pfp_n | mean_swfac_days_gt_0p05 | severe_swfac_failure_years | mean_nstres_days_gt_0p05 | unique_action_signatures | fixed_event_slots | max_slot_dap_range | candidate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 10 | 8948.1114 | 10073.4622 | 0.8953 | 4 | 225.0 | 240.0 | 37.2838 | 4.9 | 0 | 0.6 | 9 | 6 | 49 | False |
| 2000 | 10 | 7702.1549 | 10073.4622 | 0.7783 | 6 | 135.0 | 240.0 | 32.0923 | 11.2 | 2 | 0.0 | 1 | 6 | 0 | False |
| 5000 | 10 | 1515.1004 | 10073.4622 | 0.1514 | 10 | 0.0 | 0.0 |  | 22.3 | 5 | 63.8 | 1 | 0 | 0 | False |
| 10000 | 10 | 3729.7217 | 10073.4622 | 0.3805 | 10 | 0.0 | 80.0 | 46.6215 | 27.5 | 6 | 14.3 | 1 | 1 | 0 | False |
| 25000 | 10 | 9909.8435 | 10073.4622 | 0.9856 | 1 | 225.0 | 240.0 | 41.291 | 2.6 | 0 | 0.0 | 7 | 3 | 15 | True |

## 事件槽位变异性

| checkpoint_step | event_type | slot | year_count | dap_min | dap_max | dap_range | dap_std | fixed_across_years |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | irrigation | 1 | 10 | 1 | 1 | 0 | 0.0 | True |
| 1000 | irrigation | 2 | 10 | 31 | 31 | 0 | 0.0 | True |
| 1000 | irrigation | 3 | 10 | 38 | 38 | 0 | 0.0 | True |
| 1000 | irrigation | 4 | 10 | 61 | 110 | 49 | 21.4467 | False |
| 1000 | irrigation | 5 | 10 | 101 | 117 | 16 | 5.4185 | False |
| 1000 | nitrogen | 1 | 10 | 2 | 2 | 0 | 0.0 | True |
| 1000 | nitrogen | 2 | 10 | 8 | 8 | 0 | 0.0 | True |
| 1000 | nitrogen | 3 | 10 | 15 | 15 | 0 | 0.0 | True |
| 2000 | irrigation | 1 | 10 | 1 | 1 | 0 | 0.0 | True |
| 2000 | irrigation | 2 | 10 | 31 | 31 | 0 | 0.0 | True |
| 2000 | irrigation | 3 | 10 | 38 | 38 | 0 | 0.0 | True |
| 2000 | nitrogen | 1 | 10 | 2 | 2 | 0 | 0.0 | True |
| 2000 | nitrogen | 2 | 10 | 8 | 8 | 0 | 0.0 | True |
| 2000 | nitrogen | 3 | 10 | 15 | 15 | 0 | 0.0 | True |
| 10000 | nitrogen | 1 | 10 | 1 | 1 | 0 | 0.0 | True |
| 25000 | irrigation | 1 | 10 | 1 | 1 | 0 | 0.0 | True |
| 25000 | irrigation | 2 | 10 | 48 | 63 | 15 | 4.1713 | False |
| 25000 | irrigation | 3 | 10 | 55 | 70 | 15 | 4.1713 | False |
| 25000 | irrigation | 4 | 10 | 62 | 77 | 15 | 4.1713 | False |
| 25000 | irrigation | 5 | 10 | 91 | 91 | 0 | 0.0 | True |
| 25000 | nitrogen | 1 | 10 | 2 | 2 | 0 | 0.0 | True |
| 25000 | nitrogen | 2 | 10 | 49 | 64 | 15 | 4.1713 | False |
| 25000 | nitrogen | 3 | 10 | 56 | 71 | 15 | 4.1713 | False |

## 输出图

- `benchmark_results/042_14_sya_lowIC_binary_timing_checkpoint_guardrail_audit/figures/042_14_binary_timing_checkpoint_guardrail_summary.png`
- `benchmark_results/042_14_sya_lowIC_binary_timing_checkpoint_guardrail_audit/figures/042_14_binary_timing_checkpoint_guardrail_summary.svg`

## 结论边界

- 如果要把该规则正式用于下一轮训练，必须在训练前写进 prompt，而不是继续事后挑 checkpoint。
- 25K 可以作为当前候选，但它仍有固定 DAP1/DAP2/DAP91 模板成分；后续若追求更强天气响应，需要把响应性指标前置到正式选择协议中。
