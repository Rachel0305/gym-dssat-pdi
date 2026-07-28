# 035_02 FQA2014 linked MaskablePPO checkpoint guardrail 重训记录

## 结论先说

- 已训练 checkpoint 数：5。
- 已评估 checkpoint 数：5。
- guardrail 通过数：0。
- 当前 guardrail 排名第一：step 40000。
- 本任务只跑 seed0/FQA2014，不代表跨 seed、跨年、跨站点结论。

## 训练 checkpoint

| algorithm | station_code | year | seed | checkpoint_step | chunk_timesteps | run_status | model_path | task | linked_management_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | FQA | 2014 | 0 | 10000 | 10000 | ok | benchmark_results/035_02_fqa2014_linked_maskableppo_checkpoint_guardrail/models/FQA/maskableppo_linked_seed0_step10000.zip | 035_02 | True |
| MaskablePPO | FQA | 2014 | 0 | 20000 | 10000 | ok | benchmark_results/035_02_fqa2014_linked_maskableppo_checkpoint_guardrail/models/FQA/maskableppo_linked_seed0_step20000.zip | 035_02 | True |
| MaskablePPO | FQA | 2014 | 0 | 30000 | 10000 | ok | benchmark_results/035_02_fqa2014_linked_maskableppo_checkpoint_guardrail/models/FQA/maskableppo_linked_seed0_step30000.zip | 035_02 | True |
| MaskablePPO | FQA | 2014 | 0 | 40000 | 10000 | ok | benchmark_results/035_02_fqa2014_linked_maskableppo_checkpoint_guardrail/models/FQA/maskableppo_linked_seed0_step40000.zip | 035_02 | True |
| MaskablePPO | FQA | 2014 | 0 | 50000 | 10000 | ok | benchmark_results/035_02_fqa2014_linked_maskableppo_checkpoint_guardrail/models/FQA/maskableppo_linked_seed0_step50000.zip | 035_02 | True |

## checkpoint 评估

| checkpoint_step | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | simple_profit | reward_stress_aware_sum | interface_pass | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10000 | 8032.4365 | 150.0 | 240.0 | 2.17 | 33.5 | 7488.2365 | 0.7699 | True | DAP1 I15/N0; DAP2 I0/N120; DAP8 I45/N80; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| 20000 | 8076.5894 | 150.0 | 240.0 | 2.22 | 33.7 | 7532.3894 | 0.8669 | True | DAP1 I45/N120; DAP8 I45/N80; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0 |
| 30000 | 8007.522 | 150.0 | 240.0 | 2.23 | 33.4 | 7463.322 | 0.856 | True | DAP1 I45/N120; DAP8 I0/N40; DAP9 I30/N0; DAP15 I0/N40; DAP16 I30/N0; DAP22 I0/N40; DAP23 I30/N0; DAP30 I15/N0 |
| 40000 | 8088.7531 | 150.0 | 240.0 | 2.3 | 33.7 | 7544.5531 | 0.8688 | True | DAP1 I45/N40; DAP8 I30/N80; DAP15 I30/N80; DAP22 I0/N40; DAP23 I45/N0 |
| 50000 | 8032.0227 | 150.0 | 240.0 | 2.17 | 33.5 | 7487.8227 | 0.8599 | True | DAP1 I45/N40; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP30 I0/N40; DAP36 I15/N0; DAP37 I0/N40; DAP43 I15/N0; DAP50 I15/N0 |

## guardrail 排名

| guardrail_rank | checkpoint_step | guardrail_pass | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | simple_profit | delta_yield_vs_expert | delta_wp_vs_expert | delta_pfp_vs_expert | delta_profit_vs_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40000 | False | 8088.7531 | 150.0 | 240.0 | 2.3 | 33.7 | 7544.5531 | -229.6417 | -0.02 | -67.7 | -618.9817 |
| 2 | 20000 | False | 8076.5894 | 150.0 | 240.0 | 2.22 | 33.7 | 7532.3894 | -241.8054 | -0.1 | -67.7 | -631.1454 |
| 3 | 10000 | False | 8032.4365 | 150.0 | 240.0 | 2.17 | 33.5 | 7488.2365 | -285.9583 | -0.15 | -67.9 | -675.2983 |
| 4 | 50000 | False | 8032.0227 | 150.0 | 240.0 | 2.17 | 33.5 | 7487.8227 | -286.3721 | -0.15 | -67.9 | -675.7121 |
| 5 | 30000 | False | 8007.522 | 150.0 | 240.0 | 2.23 | 33.4 | 7463.322 | -310.8728 | -0.09 | -68.0 | -700.2128 |

## 解释边界

- 若没有 checkpoint 通过，本轮不能现场追加步数或调 reward。
- 若有 checkpoint 通过，只能作为 FQA2014 seed0 候选，后续仍需 seed/年份验证。
- 所有 checkpoint 均要求 linked interface_pass，否则不得作为候选。

耗时：307.4 秒。
