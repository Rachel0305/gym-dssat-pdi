# 044_01 SYA lowIC Demo-DQN Q-ranking audit 记录

## 一句话结论

分支：`B_demo_q_ranking_insufficient`。本任务不训练，只检查 044_00 Demo-DQN checkpoint 在 teacher/demo states 上的 Q 排序。

## 按 checkpoint 汇总

| checkpoint_step | sample_count | nonzero_count | teacher_argmax_rate | nonzero_teacher_argmax_rate | mean_teacher_rank | mean_q_teacher_minus_noop | mean_q_teacher_minus_max_other | margin_satisfied_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1398 | 67 | 0.9521 | 0.0 | 1.0801 | -0.0115 | 0.1981 | 0.0 |
| 2000 | 1398 | 67 | 0.9521 | 0.0 | 1.0722 | -0.0423 | 0.7373 | 0.3791 |

## 按 teacher 动作类别汇总

| checkpoint_step | teacher_action_index | sample_count | teacher_argmax_rate | mean_teacher_rank | mean_q_teacher_minus_noop | mean_q_teacher_minus_max_other | margin_satisfied_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0 | 1331 | 1.0 | 1.0 | 0.0 | 0.2343 | 0.0 |
| 1000 | 1 | 19 | 0.0 | 2.8947 | -0.2338 | -0.2338 | 0.0 |
| 1000 | 2 | 41 | 0.0 | 2.6098 | -0.2481 | -0.2481 | 0.0 |
| 1000 | 3 | 7 | 0.0 | 2.4286 | -0.215 | -0.215 | 0.0 |
| 2000 | 0 | 1331 | 1.0 | 1.0 | 0.0 | 0.8712 | 0.3982 |
| 2000 | 1 | 19 | 0.0 | 3.1053 | -0.8771 | -0.8771 | 0.0 |
| 2000 | 2 | 41 | 0.0 | 2.0732 | -0.8986 | -0.8986 | 0.0 |
| 2000 | 3 | 7 | 0.0 | 3.4286 | -0.8093 | -0.8093 | 0.0 |

## 按年份汇总

| checkpoint_step | year | sample_count | nonzero_count | teacher_argmax_rate | mean_q_teacher_minus_noop | mean_q_teacher_minus_max_other |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2014 | 152 | 8 | 0.9474 | -0.013 | 0.1865 |
| 1000 | 2015 | 141 | 6 | 0.9574 | -0.0102 | 0.2092 |
| 1000 | 2016 | 141 | 6 | 0.9574 | -0.0104 | 0.2073 |
| 1000 | 2017 | 139 | 8 | 0.9424 | -0.0136 | 0.18 |
| 1000 | 2018 | 129 | 7 | 0.9457 | -0.0135 | 0.1994 |
| 1000 | 2019 | 140 | 6 | 0.9571 | -0.0098 | 0.1874 |
| 1000 | 2020 | 138 | 7 | 0.9493 | -0.0123 | 0.1885 |
| 1000 | 2021 | 138 | 7 | 0.9493 | -0.0121 | 0.1977 |
| 1000 | 2022 | 142 | 6 | 0.9577 | -0.0107 | 0.2219 |
| 1000 | 2023 | 138 | 6 | 0.9565 | -0.0099 | 0.1903 |
| 2000 | 2014 | 152 | 8 | 0.9474 | -0.0472 | 0.6883 |
| 2000 | 2015 | 141 | 6 | 0.9574 | -0.0377 | 0.7636 |
| 2000 | 2016 | 141 | 6 | 0.9574 | -0.038 | 0.7736 |
| 2000 | 2017 | 139 | 8 | 0.9424 | -0.0483 | 0.6582 |
| 2000 | 2018 | 129 | 7 | 0.9457 | -0.0498 | 0.7561 |
| 2000 | 2019 | 140 | 6 | 0.9571 | -0.0371 | 0.7066 |
| 2000 | 2020 | 138 | 7 | 0.9493 | -0.0455 | 0.7061 |
| 2000 | 2021 | 138 | 7 | 0.9493 | -0.0443 | 0.7296 |
| 2000 | 2022 | 142 | 6 | 0.9577 | -0.0388 | 0.8293 |
| 2000 | 2023 | 138 | 6 | 0.9565 | -0.037 | 0.7176 |

## 输出文件

```json
{
  "detail": "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/tables/044_01_q_ranking_detail.csv",
  "by_checkpoint": "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/tables/044_01_q_ranking_by_checkpoint.csv",
  "by_action": "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/tables/044_01_q_ranking_by_action.csv",
  "by_year": "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/tables/044_01_q_ranking_by_year.csv",
  "demo_meta": "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/demo_buffer/044_01_demo_transition_meta.csv",
  "skipped": "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/demo_buffer/044_01_skipped_masked_teacher_actions.csv",
  "result": "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/044_01_result.json"
}
```
