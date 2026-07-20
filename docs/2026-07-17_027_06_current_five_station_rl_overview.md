# 027_06 五站点当前强化学习结果总览

## 结论先行

当前没有证据证明存在一套完全相同的参数或同一套模型权重，可使五个站点各自训练后都达到“至少一个指标严格高于另外四情景”。

阶段型 MaskablePPO 已在 SY 与 HLA 使用同一算法和同一组核心超参数；SY 明确通过产量与 WP_ET 两个全五情景可比指标，HLA 仅在正施氮情景可比的 PFP_N 上排名第一。YC/FQ/LC 尚未正式运行这套阶段型 MaskablePPO，因此不能把历史 DQN 候选当成该统一框架已经覆盖五站点的证据。

## 当前每站点代表候选

| site   |   year | representative_model   |   selected_checkpoint_step | step_unit                           |   grain_yield_kg_ha |   WP_ET_kg_m3 |   PFP_N_kg_kg |   nitrogen_total_kg_ha |   irrigation_total_mm |   max_water_stress_WSPD |   max_nitrogen_stress_NSTD |   final_soil_water_SWTD_mm | winning_metric                                       | comparable_any_metric_winner   |
|:-------|-------:|:-----------------------|---------------------------:|:------------------------------------|--------------------:|--------------:|--------------:|-----------------------:|----------------------:|------------------------:|---------------------------:|---------------------------:|:-----------------------------------------------------|:-------------------------------|
| HLA    |   2010 | MaskablePPO            |                        180 | stage transitions                   |             7853.66 |          1.69 |         157.1 |                     50 |                    90 |                   0.414 |                      0.016 |                        138 | PFP_N(comparable positive-N scenarios)               | True                           |
| YC     |   2014 | DQN                    |                       5000 | environment steps                   |             9418    |          2.53 |          37.7 |                    250 |                   120 |                   0     |                      0.013 |                        178 | none                                                 | False                          |
| FQ     |   2016 | DQN                    |                      30000 | environment steps                   |             7995    |          2.47 |         nan   |                      0 |                    60 |                   0.05  |                      0.012 |                        238 | none                                                 | False                          |
| LC     |   2010 | DQN                    |                       5000 | environment steps (smoke candidate) |             8739    |          3.06 |         nan   |                      0 |                    90 |                   0     |                      0.019 |                        252 | none                                                 | False                          |
| SY     |   2014 | MaskablePPO            |                        120 | stage transitions                   |            11204.9  |          2.31 |          56   |                    200 |                    60 |                   0.159 |                      0.136 |                        153 | yield; WP_ET; PFP_N(comparable positive-N scenarios) | True                           |

## 判定口径

- 产量和 WP_ET：候选必须严格高于另外四情景才算全五情景第一。
- PFP_N：施氮量为0时数学上未定义，不能把 NA 当作0，也不能声称严格超过四个数；仅另报在正施氮、PFP_N可定义情景中的排名。
- WSPD/NSTD 为季内最大胁迫指数，0表示无胁迫；SWTD为收获时土壤剖面水量，不是越高越好，需与胁迫和投入联合解释。
- “接近超过”尚无导师给定数值阈值，因此表中给出相对最佳其他情景的百分比差值，不擅自判定接近与否。

## 统一框架判断

1. 同一模型权重跨站点直接应用：当前证据不支持。
2. 完全相同的全部参数：当前也不成立，因为每站点至少需要自己的输入文件、IC、物候/阶段环境和观测scaler。
3. 同一算法代码和核心超参数、每站点独立训练：这是当前最可行的统一框架。SY/HLA已使用 MaskablePPO `[32,32]`、lr=3e-4、gamma=1、GAE lambda=1、n_steps=60、batch=30、n_epochs=5、240阶段步；但只验证了两个站点，尚不能称五站点通用。
4. 历史DQN五站点候选受021_05探索率日程问题影响，且YC/FQ/LC在当前五情景表中没有任何一个全五情景可比指标严格第一，不能作为通用成功框架的正式证据。

## 下一项最小实验

保持SY/HLA已经冻结的阶段型MaskablePPO算法、核心超参数和240阶段步协议不变，按027_00顺序在YC2014、FQ2016、LC2010分别做站点专属输入/scaler准备和三seed训练。只有三站也达到预注册跨seed门槛后，才能回答“同一训练框架是否适用于五站点”。在此之前不应改PPO参数或宣称已有万能参数。

## 来源

- `benchmark_results/027_05/027_05_dqn_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_ppo_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_five_scenario_summary.csv`
- `benchmark_results/027_05/027_05_daily_values.csv`
- `docs/2026-07-17_026_04_sy2014_stage_maskable_ppo_seed2_three_seed_confirmation.md`
- `docs/2026-07-17_027_03_hla2010_stage_maskable_ppo_three_seed_replication.md`
- `prompts/027_00_five_station_site_specific_stage_maskable_ppo_protocol.md`
