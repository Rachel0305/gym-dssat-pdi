# 034_04 linked 修复后自由时序 PPO/DQN 训练 smoke 记录

## 结论先说

- 接口闭环通过：2/2。
- 本任务只验证修复后训练与回放链路，不评价最终农学优劣。
- 站点年份：FQA2014；seed=0；训练步数=5000。

## 训练 summary

| algorithm | station_code | train_years | seed | total_timesteps | run_status | model_path | notes | task | linked_management_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | FQA | 2014 | 0 | 5000 | ok | benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/models/FQA/maskableppo_stress_aware_seed0.zip |  | 034_04 | True |
| DQN | FQA | 2014 | 0 | 5000 | ok | benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/models/FQA/dqn_stress_aware_seed0.zip |  | 034_04 | True |

## 评估 summary

| algorithm | run_status | episode_length | final_grnwt | total_irrigation | total_n | summary_irrigation_mm | summary_n_kg_ha | safe_summary_i_match | safe_summary_n_match | overview_is_linked | interface_pass | action_sequence | daily_csv_path | snapshot_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | ok | 114 | 8032.2229 | 150.0 | 240.0 | 150.0 | 240.0 | True | True | True | True | DAP1 I0/N120; DAP2 I15/N0; DAP8 I45/N80; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 | benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/daily_outputs/FQA/2014_maskableppo_linked_eval_daily.csv | benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/snapshots/FQA/2014/maskableppo_linked_eval |
| DQN | ok | 114 | 8060.1251 | 150.0 | 240.0 | 150.0 | 240.0 | True | True | True | True | DAP1 I15/N40; DAP8 I15/N40; DAP15 I45/N80; DAP22 I15/N40; DAP29 I15/N40; DAP36 I15/N0; DAP43 I30/N0 | benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/daily_outputs/FQA/2014_dqn_linked_eval_daily.csv | benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/snapshots/FQA/2014/dqn_linked_eval |

## 失败记录

无记录。

## 边界

- 5K 是 smoke，不是最终训练长度。
- 通过本任务只表示 linked action 已进入 DSSAT，可开始重新训练；不代表 PPO/DQN 已优于四情景。
- 033_04 旧结果仍应标注为 external action 未落地条件下的历史结果，不应用于正式比较。

耗时：75.9 秒
