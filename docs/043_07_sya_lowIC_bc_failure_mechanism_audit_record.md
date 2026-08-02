# 043_07 SYA lowIC BC failure mechanism audit 记录

## 一句话结论

分支：`B_open_loop_imitation_insufficient_and_closed_loop_template`。本任务没有训练、没有运行 DSSAT，只读取 043_06 的 BC 模型、BC dataset 和 rollout 结果。

## 开环预测汇总

| bc_epoch | sample_count | overall_accuracy | nonzero_action_accuracy | nonzero_true_count | nonzero_pred_count | nonzero_pred_rate | predicted_action_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1398 | 0.4134 | 0.5373 | 67 | 854 | 0.6109 | {"1": 656, "2": 198, "0": 544} |
| 5 | 1398 | 0.402 | 0.6269 | 67 | 878 | 0.628 | {"1": 614, "2": 264, "0": 520} |
| 20 | 1398 | 0.4041 | 0.6716 | 67 | 878 | 0.628 | {"1": 482, "2": 396, "0": 520} |
| 50 | 1398 | 0.3991 | 0.5672 | 67 | 878 | 0.628 | {"1": 656, "2": 222, "0": 520} |

## 开环按动作类别

| bc_epoch | teacher_action_index | sample_count | accuracy | predicted_counts_within_teacher_action |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1331 | 0.4072 | {"1": 608, "2": 181, "0": 542} |
| 0 | 1 | 19 | 1.0 | {"1": 19} |
| 0 | 2 | 41 | 0.4146 | {"1": 22, "2": 17, "0": 2} |
| 0 | 3 | 7 | 0.0 | {"1": 7} |
| 5 | 0 | 1331 | 0.3907 | {"1": 570, "2": 241, "0": 520} |
| 5 | 1 | 19 | 1.0 | {"1": 19} |
| 5 | 2 | 41 | 0.561 | {"1": 18, "2": 23} |
| 5 | 3 | 7 | 0.0 | {"1": 7} |
| 20 | 0 | 1331 | 0.3907 | {"1": 448, "2": 363, "0": 520} |
| 20 | 1 | 19 | 0.8947 | {"1": 17, "2": 2} |
| 20 | 2 | 41 | 0.6829 | {"1": 13, "2": 28} |
| 20 | 3 | 7 | 0.0 | {"1": 4, "2": 3} |
| 50 | 0 | 1331 | 0.3907 | {"1": 608, "2": 203, "0": 520} |
| 50 | 1 | 19 | 1.0 | {"1": 19} |
| 50 | 2 | 41 | 0.4634 | {"1": 22, "2": 19} |
| 50 | 3 | 7 | 0.0 | {"1": 7} |

## 闭环 rollout 汇总

| bc_epoch | validation_years | unique_policy_action_signatures | unique_teacher_action_signatures | exact_teacher_sequence_match_years | mean_yield | any_metric_win_years | all3_win_years | mean_policy_irrigation | mean_policy_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 2 | 8 | 0 | 9633.1 | 9 | 2 | 225.0 | 240.0 |
| 5 | 10 | 1 | 8 | 0 | 9634.2 | 9 | 2 | 225.0 | 240.0 |
| 20 | 10 | 1 | 8 | 0 | 9634.2 | 9 | 2 | 225.0 | 240.0 |
| 50 | 10 | 1 | 8 | 0 | 9634.2 | 9 | 2 | 225.0 | 240.0 |

## 闭环 teacher vs policy 示例

| bc_epoch | year | exact_sequence_match_teacher | policy_action_sequence | teacher_binary_action_sequence |
| --- | --- | --- | --- | --- |
| 0 | 2014 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I0/N80; DAP30 I45/N0; DAP43 I0/N80; DAP45 I45/N0; DAP60 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0 |
| 0 | 2015 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP31 I45/N0; DAP38 I45/N0; DAP43 I0/N80; DAP61 I45/N80; DAP91 I45/N0 |
| 0 | 2016 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP18 I45/N0; DAP25 I45/N0; DAP32 I45/N0 | DAP1 I0/N80; DAP43 I0/N80; DAP45 I45/N0; DAP75 I45/N0; DAP95 I45/N0; DAP115 I45/N0 |
| 0 | 2017 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I0/N80; DAP30 I45/N0; DAP43 I0/N80; DAP45 I45/N0; DAP60 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0 |
| 0 | 2018 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N0; DAP30 I45/N0; DAP43 I0/N80; DAP60 I45/N0; DAP61 I0/N80; DAP90 I45/N0; DAP105 I45/N0 |
| 0 | 2019 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP8 I45/N0; DAP31 I45/N0; DAP38 I45/N0; DAP43 I0/N80; DAP61 I45/N80 |
| 0 | 2020 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP30 I45/N0; DAP43 I0/N80; DAP45 I45/N0; DAP60 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0 |
| 0 | 2021 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I0/N80; DAP43 I0/N80; DAP45 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0; DAP115 I45/N0 |
| 0 | 2022 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP31 I45/N0; DAP43 I0/N80; DAP61 I45/N0; DAP91 I45/N0; DAP110 I45/N0 |
| 0 | 2023 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP8 I45/N0; DAP31 I45/N0; DAP38 I45/N0; DAP43 I0/N80; DAP61 I45/N80 |
| 5 | 2014 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I0/N80; DAP30 I45/N0; DAP43 I0/N80; DAP45 I45/N0; DAP60 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0 |
| 5 | 2015 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP31 I45/N0; DAP38 I45/N0; DAP43 I0/N80; DAP61 I45/N80; DAP91 I45/N0 |
| 5 | 2016 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I0/N80; DAP43 I0/N80; DAP45 I45/N0; DAP75 I45/N0; DAP95 I45/N0; DAP115 I45/N0 |
| 5 | 2017 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I0/N80; DAP30 I45/N0; DAP43 I0/N80; DAP45 I45/N0; DAP60 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0 |
| 5 | 2018 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N0; DAP30 I45/N0; DAP43 I0/N80; DAP60 I45/N0; DAP61 I0/N80; DAP90 I45/N0; DAP105 I45/N0 |
| 5 | 2019 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP8 I45/N0; DAP31 I45/N0; DAP38 I45/N0; DAP43 I0/N80; DAP61 I45/N80 |
| 5 | 2020 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP30 I45/N0; DAP43 I0/N80; DAP45 I45/N0; DAP60 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0 |
| 5 | 2021 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I0/N80; DAP43 I0/N80; DAP45 I45/N0; DAP61 I0/N80; DAP75 I45/N0; DAP95 I45/N0; DAP115 I45/N0 |
| 5 | 2022 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP31 I45/N0; DAP43 I0/N80; DAP61 I45/N0; DAP91 I45/N0; DAP110 I45/N0 |
| 5 | 2023 | False | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | DAP1 I45/N80; DAP8 I45/N0; DAP31 I45/N0; DAP38 I45/N0; DAP43 I0/N80; DAP61 I45/N80 |

## 输出文件

```json
{
  "open_loop_summary": "benchmark_results/043_07_sya_lowIC_bc_failure_mechanism_audit/tables/043_07_open_loop_summary.csv",
  "open_loop_by_action": "benchmark_results/043_07_sya_lowIC_bc_failure_mechanism_audit/tables/043_07_open_loop_by_action.csv",
  "open_loop_by_year": "benchmark_results/043_07_sya_lowIC_bc_failure_mechanism_audit/tables/043_07_open_loop_by_year.csv",
  "closed_loop_detail": "benchmark_results/043_07_sya_lowIC_bc_failure_mechanism_audit/tables/043_07_closed_loop_teacher_vs_policy_detail.csv",
  "closed_loop_summary": "benchmark_results/043_07_sya_lowIC_bc_failure_mechanism_audit/tables/043_07_closed_loop_summary.csv",
  "result_json": "benchmark_results/043_07_sya_lowIC_bc_failure_mechanism_audit/043_07_result.json"
}
```
