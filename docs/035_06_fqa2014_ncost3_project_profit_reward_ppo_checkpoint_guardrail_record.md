# 035_06 FQA2014 linked 自由时序 PPO ncost3 reward 记录

## 结论先说

- 训练 checkpoint 数：5；评估 checkpoint 数：5。
- guardrail 通过数：0/5。
- 本任务只改 reward 口径：训练 reward 等价于 `final_GRNWT - 1.1I - 3.0N` 后乘 0.001。
- 本任务没有调 PPO 超参数、没有增加 seed、没有扩展站点年份。

## expert 参照

| expert_yield | expert_irrigation | expert_nitrogen | expert_WP_ET | expert_PFP_N | expert_project_simple_profit |
| --- | --- | --- | --- | --- | --- |
| 8318.3948 | 23.0 | 82.0 | 2.32 | 101.4 | 8163.5348 |

## 训练 checkpoints

| algorithm | station_code | year | seed | checkpoint_step | chunk_timesteps | run_status | model_path | task | reward_type | linked_management_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | FQA | 2014 | 0 | 10000 | 10000 | ok | benchmark_results/035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail/models/FQA/maskableppo_project_profit_seed0_step10000.zip | 035_06 | harvest_ncost3_profit_scaled_0p001 | True |
| MaskablePPO | FQA | 2014 | 0 | 20000 | 10000 | ok | benchmark_results/035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail/models/FQA/maskableppo_project_profit_seed0_step20000.zip | 035_06 | harvest_ncost3_profit_scaled_0p001 | True |
| MaskablePPO | FQA | 2014 | 0 | 30000 | 10000 | ok | benchmark_results/035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail/models/FQA/maskableppo_project_profit_seed0_step30000.zip | 035_06 | harvest_ncost3_profit_scaled_0p001 | True |
| MaskablePPO | FQA | 2014 | 0 | 40000 | 10000 | ok | benchmark_results/035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail/models/FQA/maskableppo_project_profit_seed0_step40000.zip | 035_06 | harvest_ncost3_profit_scaled_0p001 | True |
| MaskablePPO | FQA | 2014 | 0 | 50000 | 10000 | ok | benchmark_results/035_06_fqa2014_ncost3_project_profit_reward_ppo_checkpoint_guardrail/models/FQA/maskableppo_project_profit_seed0_step50000.zip | 035_06 | harvest_ncost3_profit_scaled_0p001 | True |

## checkpoint guardrail 排名

| guardrail_rank | checkpoint_step | guardrail_pass | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | project_simple_profit | delta_yield_vs_expert | delta_i_vs_expert | delta_n_vs_expert | delta_wp_vs_expert | delta_pfp_vs_expert | interface_pass | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40000 | False | 7725.6079 | 150.0 | 240.0 | 2.16 | 32.2 | 7181.4079 | -592.7869 | 127.0 | 158.0 | -0.16 | -69.2 | True | DAP1 I0/N40; DAP2 I30/N0; DAP8 I0/N40; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP22 I0/N40; DAP23 I30/N0; DAP29 I0/N40; DAP30 I30/N0; DAP36 I0/N40 |
| 2 | 50000 | False | 7725.6079 | 150.0 | 240.0 | 2.16 | 32.2 | 7181.4079 | -592.7869 | 127.0 | 158.0 | -0.16 | -69.2 | True | DAP1 I0/N40; DAP2 I30/N0; DAP8 I0/N40; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP22 I0/N40; DAP23 I30/N0; DAP29 I0/N40; DAP30 I30/N0; DAP36 I0/N40 |
| 3 | 10000 | False | 7705.2185 | 150.0 | 240.0 | 2.16 | 32.1 | 7161.0185 | -613.1763 | 127.0 | 158.0 | -0.16 | -69.3 | True | DAP1 I0/N80; DAP2 I30/N0; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N80; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0 |
| 4 | 20000 | False | 7705.2185 | 150.0 | 240.0 | 2.16 | 32.1 | 7161.0185 | -613.1763 | 127.0 | 158.0 | -0.16 | -69.3 | True | DAP1 I0/N80; DAP2 I30/N0; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N80; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0 |
| 5 | 30000 | False | 7705.2185 | 150.0 | 240.0 | 2.16 | 32.1 | 7161.0185 | -613.1763 | 127.0 | 158.0 | -0.16 | -69.3 | True | DAP1 I0/N80; DAP2 I30/N0; DAP8 I0/N80; DAP9 I30/N0; DAP15 I0/N80; DAP16 I30/N0; DAP23 I30/N0; DAP30 I30/N0 |

## 边界

- 若本任务失败，只能说明该 reward 变体在 FQA2014 seed0 50K 内没有解决问题，不能推出 PPO/DQN 或自由时序整体不可行。
- 若本任务通过，也只能说明单站点单年单 seed 有正向信号，不能直接扩展到全站点。

耗时：296.0 秒

## 035_06 特别说明

- 本任务只把 nitrogen_cost 从 1.58 提高到 3.0。
- 动作空间和水氮上限没有收窄；PPO 仍可自由选择高氮，但必须用更高产量收益抵消更高氮成本。
- 排名表同时保留项目原口径 project_simple_profit 与训练同口径 ncost3_profit。
