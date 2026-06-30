# 014_02 HLA 全年份筛选与 DQN 迁移准备记录

## 本轮目标

在不重新训练、不重新运行 DSSAT 的前提下，整理 HLA 2004–2023 已有 IC=1 null/auto 诊断结果，判断海伦站是否还有值得继续做 DQN 的年份。

## 输入

- 年份筛选来源：`DSSAT_auto_validation/HLA_2004/hla_ic1_yearly_diagnostics_2004_2023/hla_original_ic1_null_vs_auto_candidate_years.csv`
- 年度日值来源：`DSSAT_auto_validation/HLA_2004/hla_ic1_yearly_diagnostics_2004_2023/hla_ic1_yearly_summary.csv`
- 旧四情景来源：`DSSAT_auto_validation/HLA_2004/hla_2010_2015_four_scenario_with_ppo/hla_2010_2015_four_scenario_summary.csv`
- HLA2010 DQN 来源：`DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/hla2010_dqn_economic_compare_summary.csv`
- HLA2015 DQN 来源：`DSSAT_auto_validation/HLA_2004/hla2010_dqn_economic_reward_probe_012_03/figures_012_05_hla2015/hla2015_economic_dqn_four_scenario_summary.csv`

## 筛选规则

- null 与 auto 均需产生非零产量。
- DSSAT auto 相对 null 增产超过 200 kg/ha，说明该年份存在低成本可见的管理响应空间。
- null 最大水分胁迫大于 0.2，避免选择几乎没有水分管理需求的年份。
- 2004、2012 这类 null 或 auto 明显失败/极端异常年份不作为第一批训练候选。

## 候选年份排序

| year | null_gwad | auto_gwad | yield_gain | auto_irrig | null_max_wspd | null_max_nstd | rain | valid_for_training_screen | exclude_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007.0 | 6002.0 | 7289.0 | 1287.0 | 143.2 | 1.0 | 0.1 | 345.6 | 1.0 | 候选 |
| 2015.0 | 6017.0 | 6988.0 | 971.0 | 141.3 | 1.0 | 0.0 | 346.0 | 1.0 | 候选 |
| 2010.0 | 6124.0 | 6936.0 | 812.0 | 189.4 | 1.0 | 0.1 | 322.2 | 1.0 | 候选 |
| 2016.0 | 6664.0 | 6934.0 | 270.0 | 140.8 | 0.7 | 0.1 | 375.8 | 1.0 | 候选 |
| 2022.0 | 6863.0 | 7132.0 | 269.0 | 142.8 | 0.8 | 0.1 | 360.4 | 1.0 | 候选 |

## 全部年份筛选表

| year | null_gwad | auto_gwad | yield_gain | auto_irrig | null_max_wspd | null_max_nstd | rain | valid_for_training_screen | exclude_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007.0 | 6002.0 | 7289.0 | 1287.0 | 143.2 | 1.0 | 0.1 | 345.6 | 1.0 | 候选 |
| 2015.0 | 6017.0 | 6988.0 | 971.0 | 141.3 | 1.0 | 0.0 | 346.0 | 1.0 | 候选 |
| 2010.0 | 6124.0 | 6936.0 | 812.0 | 189.4 | 1.0 | 0.1 | 322.2 | 1.0 | 候选 |
| 2016.0 | 6664.0 | 6934.0 | 270.0 | 140.8 | 0.7 | 0.1 | 375.8 | 1.0 | 候选 |
| 2022.0 | 6863.0 | 7132.0 | 269.0 | 142.8 | 0.8 | 0.1 | 360.4 | 1.0 | 候选 |
| 2004.0 | 0.0 | 8492.0 | 8492.0 | 377.3 | 1.0 | 0.0 | 0.0 | 0.0 | null产量为0或失败； |
| 2009.0 | 8027.0 | 8027.0 | 0.0 | 94.4 | 0.0 | 0.0 | 489.5 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2012.0 | 0.0 | 0.0 | 0.0 | 47.0 | 0.0 | 0.0 | 365.9 | 0.0 | null产量为0或失败；auto产量为0或失败；auto相对null增产不足200；null水分胁迫不明显； |
| 2017.0 | 5729.0 | 5729.0 | 0.0 | 94.5 | 0.0 | 0.0 | 392.6 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2018.0 | 6157.0 | 6157.0 | 0.0 | 47.4 | 0.0 | 0.0 | 807.3 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2023.0 | 6078.0 | 6077.0 | -1.0 | 96.0 | 0.0 | 0.0 | 662.5 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2005.0 | 7310.0 | 7308.0 | -2.0 | 47.0 | 0.0 | 0.0 | 481.8 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2006.0 | 5724.0 | 5721.0 | -3.0 | 47.4 | 0.0 | 0.0 | 533.9 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2008.0 | 7123.0 | 7119.0 | -4.0 | 46.7 | 0.0 | 0.0 | 475.9 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2014.0 | 6771.0 | 6766.0 | -5.0 | 47.4 | 0.0 | 0.0 | 543.5 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2021.0 | 5098.0 | 5089.0 | -9.0 | 47.4 | 0.0 | 0.0 | 593.2 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2020.0 | 6965.0 | 6948.0 | -17.0 | 47.4 | 0.0 | 0.0 | 778.3 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2013.0 | 5202.0 | 5183.0 | -19.0 | 96.3 | 0.0 | 0.0 | 773.7 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2019.0 | 5480.0 | 5455.0 | -25.0 | 46.5 | 0.0 | 0.0 | 612.4 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |
| 2011.0 | 6887.0 | 6835.0 | -52.0 | 93.4 | 0.0 | 0.0 | 512.1 | 0.0 | auto相对null增产不足200；null水分胁迫不明显； |

## 已有 HLA DQN/四情景结果整理

| year | source | case | yield_kg_ha | biomass_kg_ha | irrigation_mm | nitrogen_kg_ha | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2010.0 | existing_four_scenario_ppo_or_baseline | dssat_auto | 7854.0 | 20874.0 | 190.4 | 0.0 | HLA 2010/2015 旧四情景结果，PPO为旧策略/旧线索，不作为当前DQN结论 |
| 2010.0 | existing_four_scenario_ppo_or_baseline | expert_2007_shifted | 7679.0 | 20665.0 | 30.0 | 165.0 | HLA 2010/2015 旧四情景结果，PPO为旧策略/旧线索，不作为当前DQN结论 |
| 2010.0 | existing_four_scenario_ppo_or_baseline | nan | 6956.0 | 19344.0 | 0.0 | 0.0 | HLA 2010/2015 旧四情景结果，PPO为旧策略/旧线索，不作为当前DQN结论 |
| 2015.0 | existing_four_scenario_ppo_or_baseline | dssat_auto | 7648.0 | 19021.0 | 141.5 | 0.0 | HLA 2010/2015 旧四情景结果，PPO为旧策略/旧线索，不作为当前DQN结论 |
| 2015.0 | existing_four_scenario_ppo_or_baseline | expert_2007_shifted | 7296.0 | 18639.0 | 30.0 | 165.0 | HLA 2010/2015 旧四情景结果，PPO为旧策略/旧线索，不作为当前DQN结论 |
| 2015.0 | existing_four_scenario_ppo_or_baseline | nan | 6486.0 | 17168.0 | 0.0 | 0.0 | HLA 2010/2015 旧四情景结果，PPO为旧策略/旧线索，不作为当前DQN结论 |
| 2010.0 | existing_hla2010_economic_dqn_012_03 | PPO_windowed_5K |  |  | 120.1 | 150.0 | 已有HLA2010经济奖励DQN；seed0 5K曾达到DSSAT auto同量级，但仍需统一新流程复核 |
| 2010.0 | existing_hla2010_economic_dqn_012_03 | DQN_original_5K | 7373.0 | 20159.7 | 30.0 | 150.0 | 已有HLA2010经济奖励DQN；seed0 5K曾达到DSSAT auto同量级，但仍需统一新流程复核 |
| 2010.0 | existing_hla2010_economic_dqn_012_03 | DQN_economic_200 | 7532.0 | 20412.7 | 30.0 | 50.0 | 已有HLA2010经济奖励DQN；seed0 5K曾达到DSSAT auto同量级，但仍需统一新流程复核 |
| 2010.0 | existing_hla2010_economic_dqn_012_03 | DQN_economic_5K | 7854.0 | 20858.0 | 120.0 | 50.0 | 已有HLA2010经济奖励DQN；seed0 5K曾达到DSSAT auto同量级，但仍需统一新流程复核 |
| 2015.0 | existing_hla2015_economic_dqn_012_05 | dqn_economic_seed0 | 6485.8 | 17167.8 | 0.0 | 0.0 | 已有HLA2015经济DQN结果：seed0未优于null，说明2015当前设置下不稳 |
| 2015.0 | existing_hla2015_economic_dqn_012_05 | dssat_auto | 7648.0 | 19021.0 | 141.5 | 0.0 | 已有HLA2015经济DQN结果：seed0未优于null，说明2015当前设置下不稳 |
| 2015.0 | existing_hla2015_economic_dqn_012_05 | expert_2007_shifted | 7296.0 | 18639.0 | 30.0 | 165.0 | 已有HLA2015经济DQN结果：seed0未优于null，说明2015当前设置下不稳 |
| 2015.0 | existing_hla2015_economic_dqn_012_05 | null_zero | 6486.0 | 17168.0 | 0.0 | 0.0 | 已有HLA2015经济DQN结果：seed0未优于null，说明2015当前设置下不稳 |

## 初步结论

- HLA 仍然有可探索年份，但不建议回到 2004；2004 null 产量为 0，太极端，不适合作为第一批 DQN 训练验证。
- 按已有 IC=1 null/auto 诊断，优先候选是 2007、2015、2010，其次是 2016、2022。
- 已有 HLA2010 经济奖励 DQN seed0 5K 曾达到 DSSAT auto 同量级，并且用水更少、施氮 50 kg/ha；这说明 HLA 不是完全没有希望。
- 已有 HLA2015 经济奖励 DQN seed0 结果接近 null，说明 2015 对当前 DQN 设置不稳定，不能直接当成功案例。
- 下一步如果继续 HLA，建议先选择 2010 做统一新流程复核；如果 2010 复核成立，再跑 2007 或 2016，而不是直接多年份大训练。

## 图件

- `DSSAT_auto_validation/HLA_2004/hla_all_year_screen_and_dqn_transfer_014_02/figures/014_02_hla_null_auto_yield_gain.png`

## 输出文件

- `DSSAT_auto_validation/HLA_2004/hla_all_year_screen_and_dqn_transfer_014_02/014_02_hla_candidate_year_screen_summary.csv`
- `DSSAT_auto_validation/HLA_2004/hla_all_year_screen_and_dqn_transfer_014_02/014_02_hla_existing_dqn_summary.csv`
