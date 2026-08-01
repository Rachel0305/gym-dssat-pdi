# 042_05 SYA lowIC 042_02 checkpoint 天气响应性 guardrail 审计记录

## 任务性质

- 只读取 042_02 rerun100k 的 daily/evaluation CSV。
- 不训练 PPO，不运行 DSSAT，不修改 checkpoint。
- 目的：把“终点指标好”和“动作是否根据天气/胁迫变化”拆开审计。

## 预注册 guardrail

- 终点指标：`any_metric_win_years == 10` 且 `all3_win_years >= 5`。
- 非模板化：`unique_nonzero_action_signatures >= 3`。
- 灌溉响应：`mean_irrigation_event_dap_sd >= 3` 或 `irrigation_future7_rain_abs_std_diff >= 0.2`。
- 施肥响应：`mean_nitrogen_event_dap_sd >= 3` 或 `nitrogen_nstres_abs_std_diff >= 0.2`。

## checkpoint guardrail 总表

| checkpoint_step | any_metric_win_years | all3_win_years | unique_nonzero_action_signatures | mean_irrigation_event_dap_sd | irrigation_future7_rain_abs_std_diff | mean_nitrogen_event_dap_sd | nitrogen_nstres_abs_std_diff | overall_response_guardrail_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 2 | 1 | 0.0 | 0.2006 | 0.0 | 0.0522 | False |
| 25000 | 0 | 0 | 1 | 0.0 | 0.1613 | 0.0 |  | False |
| 50000 | 0 | 0 | 1 | 0.0 | 0.1613 | 0.0 |  | False |
| 75000 | 10 | 6 | 1 | 0.0 | 0.1613 | 0.0 | 0.0651 | False |
| 100000 | 0 | 0 | 1 | 0.0 | 0.1613 | 0.0 |  | False |

## 75K checkpoint 单独摘录

| checkpoint_step | unique_nonzero_action_signatures | unique_irrigation_signatures | unique_nitrogen_signatures | sd_total_irrigation | sd_total_nitrogen | sd_irrigation_events | sd_nitrogen_events | mean_irrigation_event_dap_sd | mean_nitrogen_event_dap_sd | irrigation_future7_rain_abs_std_diff | irrigation_past7_rain_abs_std_diff | irrigation_swfac_abs_std_diff | nitrogen_nstres_abs_std_diff | nitrogen_dap_abs_std_diff | years | mean_yield | mean_wp_et | mean_pfp_n | mean_irrigation | mean_nitrogen | any_metric_win_years | all3_win_years | max_swfac | max_nstres | endpoint_guardrail_pass | non_template_guardrail_pass | irrigation_response_guardrail_pass | nitrogen_response_guardrail_pass | overall_response_guardrail_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 75000 | 1 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1613 | 0.1848 | 0.1106 | 0.0651 | 1.502 | 10 | 9931.7 | 2.145 | 41.36 | 225.0 | 240.0 | 10 | 6 | 0.8592 | 0.238 | True | False | False | False | False |

## 判定

- 分支：`C_no_checkpoint_passes_endpoint_plus_response_guardrail`
- 通过 checkpoint：[]
- 若分支为 C，说明当前 042_02 虽有高指标 checkpoint，但没有 checkpoint 同时满足“高指标 + 非模板化天气响应”审计门槛。
- 下一步应改 checkpoint selection guardrail 或训练目标，使策略不仅终点指标好，而且不同年份的管理动作能随天气/胁迫输入变化。

## 输出文件

- guardrail summary: `benchmark_results\042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail\tables\042_05_checkpoint_response_guardrail_summary.csv`
- action sequence: `benchmark_results\042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail\tables\042_05_action_sequence_by_year_checkpoint.csv`
- event variability: `benchmark_results\042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail\tables\042_05_event_position_variability.csv`
- feature association: `benchmark_results\042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail\tables\042_05_action_feature_association.csv`
- figure: `benchmark_results\042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail\figures\042_05_checkpoint_metric_response_guardrail.png`
