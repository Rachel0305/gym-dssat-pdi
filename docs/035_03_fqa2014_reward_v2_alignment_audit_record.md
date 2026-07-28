# 035_03 FQA2014 自由时序 reward v2 对齐审计记录

## 结论先说

- 当前 logged reward 排名第一是 `noop`，这不是我们希望强化学习优先学习的节水节氮高产策略。
- 项目现有 `simple_profit = yield - 1.1 * irrigation - 1.58 * nitrogen` 排名第一是 `water_saving_n160`。
- 重新计算的项目 simple_profit 与 035_00 CSV 原列最大差异为 `0`，说明本次公式和既有结果口径一致。
- 在 035_00 已有候选中，项目 simple_profit 及其产量门槛变体能把 water_saving_n160 / split_moderate_n160 / critical_i90_n200 排到前列。
- 旧代理 `Y - I - 5N` 会把 no-op 排在第一，不适合作为当前 FQA2014 自由时序训练主目标。
- 因此 035_02 的失败更像是当前 logged reward 与最终指标不对齐，而不是单纯训练步数不够。
- 本任务不启动训练；若继续，应另开任务测试项目 simple_profit 或其带产量 guardrail 的变体。

## expert 参照值

| expert_yield | expert_irrigation | expert_nitrogen | expert_WP_ET | expert_PFP_N | expert_simple_profit |
| --- | --- | --- | --- | --- | --- |
| 8318.3948 | 23.0 | 82.0 | 2.32 | 101.4 | 8163.5348 |

## 各候选分数的一句话审计

| score_name | top1 | top1_one_metric_pass | top3_names | bad_candidate_in_top3 |
| --- | --- | --- | --- | --- |
| logged_reward_sum | noop | False | noop; stress_triggered; water_saving_n160 | True |
| score_project_simple_profit | water_saving_n160 | True | water_saving_n160; critical_i90_n200; split_moderate_n160 | False |
| score_old_proxy_y_i_5n | noop | False | noop; stress_triggered; water_saving_n160 | True |
| score_yield_gate_project_profit_1620 | water_saving_n160 | True | water_saving_n160; critical_i90_n200; split_moderate_n160 | False |
| score_yield_shortfall_project_penalty_x5 | water_saving_n160 | True | water_saving_n160; critical_i90_n200; split_moderate_n160 | False |

## 候选策略完整分数表

| name | grain_yield_kg_ha | irrigation_mm | nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | logged_reward_sum | score_project_simple_profit | score_old_proxy_y_i_5n | score_yield_gate_project_profit_1620 | score_yield_shortfall_project_penalty_x5 | delta_yield_vs_expert | delta_i_vs_expert | delta_n_vs_expert | advisor_one_metric_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| water_saving_n160 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | 1.1042 | 8315.0785 | 7772.3785 | 9935.0785 | 8315.0785 | 298.9838 | 22.0 | 78.0 | True |
| critical_i90_n200 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | 0.9996 | 8253.5272 | 7578.5272 | 9873.5272 | 8253.5272 | 350.1324 | 67.0 | 118.0 | True |
| split_moderate_n160 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | 1.0665 | 8252.009 | 7712.309 | 9872.009 | 8252.009 | 268.9142 | 52.0 | 78.0 | True |
| delayed_late | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | 0.8635 | 8189.414 | 7380.614 | 9809.414 | 8189.414 | 382.2192 | 97.0 | 158.0 | True |
| stress_triggered | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | 1.2342 | 8090.9467 | 7820.3467 | 8090.9467 | 7750.7062 | -68.0481 | 7.0 | -2.0 | True |
| noop | 7953.631 | 0.0 | 0.0 | 2.26 |  | 1.2567 | 7953.631 | 7953.631 | 7953.631 | 6129.812 | -364.7638 | -23.0 | -82.0 | False |
| early_dump_cap | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | 0.8782 | 7604.1105 | 6798.3105 | 7604.1105 | 6753.6894 | -170.0842 | 127.0 | 158.0 | False |
| ppo_03405_replay | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | 0.8675 | 7535.927 | 6730.127 | 7535.927 | 6344.5878 | -238.2678 | 127.0 | 158.0 | False |

## 按分数展开的排名

| score_name | rank | name | score_value | grain_yield_kg_ha | irrigation_mm | nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | advisor_one_metric_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logged_reward_sum | 1 | noop | 1.2567 | 7953.631 | 0.0 | 0.0 | 2.26 |  | False |
| logged_reward_sum | 2 | stress_triggered | 1.2342 | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | True |
| logged_reward_sum | 3 | water_saving_n160 | 1.1042 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | True |
| logged_reward_sum | 4 | split_moderate_n160 | 1.0665 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | True |
| logged_reward_sum | 5 | critical_i90_n200 | 0.9996 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | True |
| logged_reward_sum | 6 | early_dump_cap | 0.8782 | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | False |
| logged_reward_sum | 7 | ppo_03405_replay | 0.8675 | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | False |
| logged_reward_sum | 8 | delayed_late | 0.8635 | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | True |
| score_project_simple_profit | 1 | water_saving_n160 | 8315.0785 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | True |
| score_project_simple_profit | 2 | critical_i90_n200 | 8253.5272 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | True |
| score_project_simple_profit | 3 | split_moderate_n160 | 8252.009 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | True |
| score_project_simple_profit | 4 | delayed_late | 8189.414 | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | True |
| score_project_simple_profit | 5 | stress_triggered | 8090.9467 | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | True |
| score_project_simple_profit | 6 | noop | 7953.631 | 7953.631 | 0.0 | 0.0 | 2.26 |  | False |
| score_project_simple_profit | 7 | early_dump_cap | 7604.1105 | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | False |
| score_project_simple_profit | 8 | ppo_03405_replay | 7535.927 | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | False |
| score_old_proxy_y_i_5n | 1 | noop | 7953.631 | 7953.631 | 0.0 | 0.0 | 2.26 |  | False |
| score_old_proxy_y_i_5n | 2 | stress_triggered | 7820.3467 | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | True |
| score_old_proxy_y_i_5n | 3 | water_saving_n160 | 7772.3785 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | True |
| score_old_proxy_y_i_5n | 4 | split_moderate_n160 | 7712.309 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | True |
| score_old_proxy_y_i_5n | 5 | critical_i90_n200 | 7578.5272 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | True |
| score_old_proxy_y_i_5n | 6 | delayed_late | 7380.614 | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | True |
| score_old_proxy_y_i_5n | 7 | early_dump_cap | 6798.3105 | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | False |
| score_old_proxy_y_i_5n | 8 | ppo_03405_replay | 6730.127 | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | False |
| score_yield_gate_project_profit_1620 | 1 | water_saving_n160 | 9935.0785 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | True |
| score_yield_gate_project_profit_1620 | 2 | critical_i90_n200 | 9873.5272 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | True |
| score_yield_gate_project_profit_1620 | 3 | split_moderate_n160 | 9872.009 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | True |
| score_yield_gate_project_profit_1620 | 4 | delayed_late | 9809.414 | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | True |
| score_yield_gate_project_profit_1620 | 5 | stress_triggered | 8090.9467 | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | True |
| score_yield_gate_project_profit_1620 | 6 | noop | 7953.631 | 7953.631 | 0.0 | 0.0 | 2.26 |  | False |
| score_yield_gate_project_profit_1620 | 7 | early_dump_cap | 7604.1105 | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | False |
| score_yield_gate_project_profit_1620 | 8 | ppo_03405_replay | 7535.927 | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | False |
| score_yield_shortfall_project_penalty_x5 | 1 | water_saving_n160 | 8315.0785 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | True |
| score_yield_shortfall_project_penalty_x5 | 2 | critical_i90_n200 | 8253.5272 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | True |
| score_yield_shortfall_project_penalty_x5 | 3 | split_moderate_n160 | 8252.009 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | True |
| score_yield_shortfall_project_penalty_x5 | 4 | delayed_late | 8189.414 | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | True |
| score_yield_shortfall_project_penalty_x5 | 5 | stress_triggered | 7750.7062 | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | True |
| score_yield_shortfall_project_penalty_x5 | 6 | early_dump_cap | 6753.6894 | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | False |
| score_yield_shortfall_project_penalty_x5 | 7 | ppo_03405_replay | 6344.5878 | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | False |
| score_yield_shortfall_project_penalty_x5 | 8 | noop | 6129.812 | 7953.631 | 0.0 | 0.0 | 2.26 |  | False |

## 下一步建议

035_04 不应继续沿用当前 logged reward 直接加步数。建议只改 reward / 选模目标，优先测试：

1. 训练即时 reward 改为与项目综合指标一致的 `delta_GRNWT - 1.1I - 1.58N`；
2. checkpoint 选择使用统一 DSSAT 回放后的项目 simple_profit 与 expert guardrail；
3. 仍先只做 FQA2014 单站点单年，不扩展全站点；
4. 若仍打满 I150/N240，再判断是 reward 传播问题还是动作空间/约束仍不足。
