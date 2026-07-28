# 035_00 FQA2014 linked 自由时序规则探针记录

## 结论先说

- 规则策略接口闭环通过：8/8。
- 本任务不训练 PPO/DQN，只在 linked 修复后的真实动作链路下测试固定规则策略。
- 当前 simple_profit 最高规则：water_saving_n160。
- 该结果用于指导下一轮 RL 训练目标，不是最终管理方案。

## 规则策略 summary

| rule | grain_yield_kg_ha | summary_irrigation_mm | summary_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | simple_profit | reward_sum | interface_pass | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noop | 7953.631 | 0.0 | 0.0 | 2.26 |  | 7953.631 | 1.2567 | True |  |
| early_dump_cap | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | 7604.1105 | 0.8782 | True | DAP1 I45/N120; DAP8 I45/N120; DAP15 I45/N0; DAP22 I15/N0 |
| ppo_03405_replay | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | 7535.927 | 0.8675 | True | DAP1 I45/N40; DAP8 I45/N120; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 |
| split_moderate_n160 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | 8252.009 | 1.0665 | True | DAP1 I15/N40; DAP30 I15/N40; DAP50 I15/N40; DAP65 I15/N40; DAP85 I15/N0 |
| critical_i90_n200 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | 8253.5272 | 0.9996 | True | DAP1 I15/N40; DAP30 I15/N40; DAP50 I30/N80; DAP65 I30/N40 |
| water_saving_n160 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | 8315.0785 | 1.1042 | True | DAP1 I15/N40; DAP30 I0/N40; DAP50 I15/N40; DAP65 I15/N40 |
| delayed_late | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | 8189.414 | 0.8635 | True | DAP50 I30/N80; DAP65 I30/N80; DAP80 I30/N80; DAP95 I30/N0 |
| stress_triggered | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | 8090.9467 | 1.2342 | True | DAP1 I30/N0; DAP87 I0/N80 |

## 与四情景基线及 034_05 PPO 50K 对比

| source | name | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | simple_profit | delta_vs_best_baseline_grain_yield_kg_ha | delta_vs_best_baseline_WP_ET_kg_m3 | delta_vs_best_baseline_PFP_N_kg_kg | delta_vs_best_baseline_simple_profit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_03400 | dssat_auto | 7953.631 | 0.0 | 0.0 | 2.26 |  | 7953.631 | -364.7638 | -0.06 |  | -209.9038 |
| baseline_03400 | null | 7953.631 | 0.0 | 0.0 | 2.26 |  | 7953.631 | -364.7638 | -0.06 |  | -209.9038 |
| baseline_03400 | official_extension_expert | 8318.3948 | 23.0 | 82.0 | 2.32 | 101.4 | 8163.5348 | 0.0 | 0.0 | 0.0 | 0.0 |
| baseline_03400 | recorded_farmer_template | 8273.1494 | 75.0 | 144.0 | 2.27 | 57.5 | 7963.1294 | -45.2454 | -0.05 | -43.9 | -200.4054 |
| linked_rule_03500 | critical_i90_n200 | 8668.5272 | 90.0 | 200.0 | 2.41 | 43.3 | 8253.5272 | 350.1324 | 0.09 | -58.1 | 89.9924 |
| linked_rule_03500 | delayed_late | 8700.614 | 120.0 | 240.0 | 2.47 | 36.3 | 8189.414 | 382.2192 | 0.15 | -65.1 | 25.8792 |
| linked_rule_03500 | early_dump_cap | 8148.3105 | 150.0 | 240.0 | 2.29 | 34.0 | 7604.1105 | -170.0842 | -0.03 | -67.4 | -559.4242 |
| linked_rule_03500 | noop | 7953.631 | 0.0 | 0.0 | 2.26 |  | 7953.631 | -364.7638 | -0.06 |  | -209.9038 |
| linked_rule_03500 | ppo_03405_replay | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | 7535.927 | -238.2678 | -0.1 | -67.7 | -627.6078 |
| linked_rule_03500 | split_moderate_n160 | 8587.309 | 75.0 | 160.0 | 2.39 | 53.7 | 8252.009 | 268.9142 | 0.07 | -47.7 | 88.4742 |
| linked_rule_03500 | stress_triggered | 8250.3467 | 30.0 | 80.0 | 2.32 | 103.1 | 8090.9467 | -68.0481 | 0.0 | 1.7 | -72.5881 |
| linked_rule_03500 | water_saving_n160 | 8617.3785 | 45.0 | 160.0 | 2.42 | 53.9 | 8315.0785 | 298.9838 | 0.1 | -47.5 | 151.5438 |
| rl_03405 | linked_free_timing_maskableppo_50k | 8080.127 | 150.0 | 240.0 | 2.22 | 33.7 | 7535.927 | -238.2678 | -0.1 | -67.7 | -627.6078 |

## 解释边界

- 若某个规则优于 PPO 50K，说明当前 PPO 尚未学到该可达行为，不等于规则是最终答案。
- 若某个规则超过 expert，说明 linked 自由时序动作空间存在可达优质策略，下一步应训练 PPO/DQN 学到它。
- 若所有规则都不超过 expert，也不能直接判死 RL，只能说明本轮规则集合未发现足够好候选。

## 输出文件

- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_rule_summary.csv`
- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/evaluation/035_00_scenario_comparison.csv`
- `benchmark_results/035_00_fqa2014_linked_free_timing_rule_probe/daily_outputs/FQA/*.csv`

耗时：29.0 秒。
