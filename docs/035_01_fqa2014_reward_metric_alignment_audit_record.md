# 035_01 FQA2014 reward-指标对齐审计记录

## 结论先说

- 当前 reward 排名第一：noop；simple_profit 排名第一：water_saving_n160。
- 034_05 PPO 50K 的 reward 排名：7；simple_profit 排名：8。
- 当前 reward 与最终指标并不等价；只用 reward 选 checkpoint 会有明显风险。
- PPO 50K 既不是 reward 最优，也不是指标最优；下一步要同时处理训练优化不足和选择指标错位。

## reward 排序表

| source | name | reward_sum | rank_reward_sum | grain_yield_kg_ha | rank_grain_yield_kg_ha | WP_ET_kg_m3 | rank_WP_ET_kg_m3 | PFP_N_kg_kg | rank_PFP_N_kg_kg | simple_profit | rank_simple_profit | summary_irrigation_mm | summary_nitrogen_kg_ha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linked_rule_03500 | noop | 1.2567 | 1.0 | 7953.631 | 9.0 | 2.26 | 7.0 |  |  | 7953.631 | 6.0 | 0.0 | 0.0 |
| linked_rule_03500 | stress_triggered | 1.2342 | 2.0 | 8250.3467 | 5.0 | 2.32 | 5.0 | 103.1 | 1.0 | 8090.9467 | 5.0 | 30.0 | 80.0 |
| linked_rule_03500 | water_saving_n160 | 1.1042 | 3.0 | 8617.3785 | 3.0 | 2.42 | 2.0 | 53.9 | 2.0 | 8315.0785 | 1.0 | 45.0 | 160.0 |
| linked_rule_03500 | split_moderate_n160 | 1.0665 | 4.0 | 8587.309 | 4.0 | 2.39 | 4.0 | 53.7 | 3.0 | 8252.009 | 3.0 | 75.0 | 160.0 |
| linked_rule_03500 | critical_i90_n200 | 0.9996 | 5.0 | 8668.5272 | 2.0 | 2.41 | 3.0 | 43.3 | 4.0 | 8253.5272 | 2.0 | 90.0 | 200.0 |
| linked_rule_03500 | early_dump_cap | 0.8782 | 6.0 | 8148.3105 | 6.0 | 2.29 | 6.0 | 34.0 | 6.0 | 7604.1105 | 7.0 | 150.0 | 240.0 |
| linked_rule_03500 | ppo_03405_replay | 0.8675 | 7.0 | 8080.127 | 7.0 | 2.22 | 8.0 | 33.7 | 7.0 | 7535.927 | 8.0 | 150.0 | 240.0 |
| rl_03405 | linked_free_timing_maskableppo_50k | 0.8675 | 7.0 | 8080.127 | 7.0 | 2.22 | 8.0 | 33.6672 | 8.0 | 7535.927 | 8.0 | 150.0 | 240.0 |
| linked_rule_03500 | delayed_late | 0.8635 | 9.0 | 8700.614 | 1.0 | 2.47 | 1.0 | 36.3 | 5.0 | 8189.414 | 4.0 | 120.0 | 240.0 |

## reward 与指标相关性

| metric | spearman_vs_reward | pearson_vs_reward | n_non_na |
| --- | --- | --- | --- |
| grain_yield_kg_ha | -0.2269 | -0.0856 | 9 |
| WP_ET_kg_m3 | 0.0252 | 0.0637 | 9 |
| PFP_N_kg_kg | 0.8503 | 0.918 | 8 |
| simple_profit | 0.3613 | 0.5015 | 9 |
| summary_irrigation_mm | -0.8852 | -0.9733 | 9 |
| summary_nitrogen_kg_ha | -0.9571 | -0.9586 | 9 |

## 相对 expert 差值

| name | delta_yield_vs_expert | delta_i_vs_expert | delta_n_vs_expert | delta_wp_vs_expert | delta_pfp_vs_expert | delta_profit_vs_expert | advisor_one_metric_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| noop | -364.7638 | -23.0 | -82.0 | -0.06 |  | -209.9038 | False |
| stress_triggered | -68.0481 | 7.0 | -2.0 | 0.0 | 1.7 | -72.5881 | True |
| water_saving_n160 | 298.9838 | 22.0 | 78.0 | 0.1 | -47.5 | 151.5438 | True |
| split_moderate_n160 | 268.9142 | 52.0 | 78.0 | 0.07 | -47.7 | 88.4742 | True |
| critical_i90_n200 | 350.1324 | 67.0 | 118.0 | 0.09 | -58.1 | 89.9924 | True |
| early_dump_cap | -170.0842 | 127.0 | 158.0 | -0.03 | -67.4 | -559.4242 | False |
| ppo_03405_replay | -238.2678 | 127.0 | 158.0 | -0.1 | -67.7 | -627.6078 | False |
| linked_free_timing_maskableppo_50k | -238.2678 | 127.0 | 158.0 | -0.1 | -67.7328 | -627.6078 | False |
| delayed_late | 382.2192 | 97.0 | 158.0 | 0.15 | -65.1 | 25.8792 | True |

## guardrail 候选排序

| name | guardrail_score | grain_yield_kg_ha | summary_irrigation_mm | summary_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | simple_profit | reward_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| water_saving_n160 | 3 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | 8315.0785 | 1.1042 |
| critical_i90_n200 | 3 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | 8253.5272 | 0.9996 |
| split_moderate_n160 | 3 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | 8252.009 | 1.0665 |
| delayed_late | 3 | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | 8189.414 | 0.8635 |
| stress_triggered | 1 | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | 8090.9467 | 1.2342 |
| noop | 0 | 7953.631 | 0.0 | 0.0 | 2.26 |  | 7953.631 | 1.2567 |
| early_dump_cap | 0 | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | 7604.1105 | 0.8782 |
| ppo_03405_replay | 0 | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | 7535.927 | 0.8675 |
| linked_free_timing_maskableppo_50k | 0 | 8080.127 | 150.0 | 240.0 | 2.22 | 33.6672 | 7535.927 | 0.8675 |

## 下一步建议

1. 不要继续只按当前 reward 选择 checkpoint。
2. 下一轮 linked PPO/DQN 训练必须保存中间 checkpoint，并用外部指标 guardrail 复评。
3. 最低 guardrail 建议：产量不低于 expert，且产量/WP_ET/PFP_N 至少一个超过 expert；同分时优先 simple_profit。
4. 若模型训练 reward 高但不满足 guardrail，不应作为成功策略。
5. 若模型无法产生 guardrail 候选，应优先改 reward 或训练稳定性，而不是扩大全站点。

