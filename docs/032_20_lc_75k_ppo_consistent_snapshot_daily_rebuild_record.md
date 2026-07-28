# 032_20 LC 75k PPO 一致 DSSAT snapshot 日过程重建记录

## 结论先说

- 本轮没有训练、没有重新选择 checkpoint；只对已冻结 `75000` 步模型做确定性评估并保存 snapshot。
- LC2005-LC2020 共 `16` 年；五情景来源完整年份数：`16/16`。
- PPO 候选 snapshot 生成/复用状态：`{'ok_existing': 16}`。
- QA 未通过条目数：`0`。

## 为什么做这一轮

032_19 发现旧日值表存在 PPO 候选重复 date-DAP 行和五情景降雨总量不一致问题，因此旧图中的 WSPD/NSTD 过程线不能直接用于解释 PPO 决策。本轮从 DSSAT 原始输出重新解析日过程，目的是先修正证据口径，而不是改善模型结果。

## 模型与边界

- 模型：`benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt75000.zip`。
- 算法：LC 多年自由时序 stress-aware MaskablePPO。
- 训练年份：LC2005-LC2010；本轮只做冻结评估。
- 验证/绘图年份：LC2005-LC2020。
- 不调用 `learn()`；不修改原始输入；不覆盖 032_17。

## Snapshot 来源清单

| year | scenario | source | required_files_complete | snapshot_path |
| --- | --- | --- | --- | --- |
| 2005 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2005/null |
| 2005 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2005/recorded_farmer_template_02705 |
| 2005 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2005/dssat_auto |
| 2005 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2005/official_extension_expert |
| 2005 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2005/rl_candidate |
| 2006 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2006/null |
| 2006 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2006/recorded_farmer_template_02705 |
| 2006 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2006/dssat_auto |
| 2006 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2006/official_extension_expert |
| 2006 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2006/rl_candidate |
| 2007 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2007/null |
| 2007 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2007/recorded_farmer_template_02705 |
| 2007 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2007/dssat_auto |
| 2007 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2007/official_extension_expert |
| 2007 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2007/rl_candidate |
| 2008 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2008/null |
| 2008 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2008/recorded_farmer_template_02705 |
| 2008 | dssat_auto | 032_18_completed_dssat_auto_snapshot | True | benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/snapshots/LCA/2008/dssat_auto |
| 2008 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2008/official_extension_expert |
| 2008 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2008/rl_candidate |
| 2009 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2009/null |
| 2009 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2009/recorded_farmer_template_02705 |
| 2009 | dssat_auto | 032_18_completed_dssat_auto_snapshot | True | benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/snapshots/LCA/2009/dssat_auto |
| 2009 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2009/official_extension_expert |
| 2009 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2009/rl_candidate |
| 2010 | null | 032_20_local_lc2010_regenerated_baseline_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2010/null |
| 2010 | recorded_farmer | 032_20_local_lc2010_regenerated_baseline_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2010/recorded_farmer |
| 2010 | dssat_auto | 032_20_local_lc2010_regenerated_baseline_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2010/dssat_auto |
| 2010 | official_extension_expert | 032_20_local_lc2010_regenerated_baseline_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2010/official_extension_expert |
| 2010 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2010/rl_candidate |
| 2011 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2011/null |
| 2011 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2011/recorded_farmer_template_02705 |
| 2011 | dssat_auto | 032_18_completed_dssat_auto_snapshot | True | benchmark_results/032_18_lc_missing_dssat_auto_daily_completion/snapshots/LCA/2011/dssat_auto |
| 2011 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2011/official_extension_expert |
| 2011 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2011/rl_candidate |
| 2012 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2012/null |
| 2012 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2012/recorded_farmer_template_02705 |
| 2012 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2012/dssat_auto |
| 2012 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2012/official_extension_expert |
| 2012 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2012/rl_candidate |
| 2013 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2013/null |
| 2013 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2013/recorded_farmer_template_02705 |
| 2013 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2013/dssat_auto |
| 2013 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2013/official_extension_expert |
| 2013 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2013/rl_candidate |
| 2014 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2014/null |
| 2014 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2014/recorded_farmer_template_02705 |
| 2014 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2014/dssat_auto |
| 2014 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2014/official_extension_expert |
| 2014 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2014/rl_candidate |
| 2015 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2015/null |
| 2015 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2015/recorded_farmer_template_02705 |
| 2015 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2015/dssat_auto |
| 2015 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2015/official_extension_expert |
| 2015 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2015/rl_candidate |
| 2016 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2016/null |
| 2016 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2016/recorded_farmer_template_02705 |
| 2016 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2016/dssat_auto |
| 2016 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2016/official_extension_expert |
| 2016 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2016/rl_candidate |
| 2017 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2017/null |
| 2017 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2017/recorded_farmer_template_02705 |
| 2017 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2017/dssat_auto |
| 2017 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2017/official_extension_expert |
| 2017 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2017/rl_candidate |
| 2018 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2018/null |
| 2018 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2018/recorded_farmer_template_02705 |
| 2018 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2018/dssat_auto |
| 2018 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2018/official_extension_expert |
| 2018 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2018/rl_candidate |
| 2019 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2019/null |
| 2019 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2019/recorded_farmer_template_02705 |
| 2019 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2019/dssat_auto |
| 2019 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2019/official_extension_expert |
| 2019 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2019/rl_candidate |
| 2020 | null | 031_35_null_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2020/null |
| 2020 | recorded_farmer | 031_35_recorded_farmer_template_02705_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2020/recorded_farmer_template_02705 |
| 2020 | dssat_auto | 031_36_completed_dssat_auto_snapshot | True | benchmark_results/031_36_missing_dssat_auto_completion_for_03134/snapshots/LCA/2020/dssat_auto |
| 2020 | official_extension_expert | 031_35_official_extension_expert_snapshot | True | benchmark_results/031_35_missing_four_baseline_completion_for_03134/snapshots/LCA/2020/official_extension_expert |
| 2020 | rl_candidate | 032_20_frozen_ppo_reevaluation_snapshot | True | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/snapshots/LCA/2020/rl_candidate |

## LC2010 特别说明

第一次重建时，LC2010 使用 027_05 历史 snapshot 造成五情景降雨总量不一致，因此本轮在 032_20 输出目录内重新生成 LC2010 的 null、recorded_farmer、dssat_auto、official_extension_expert 四条基线 snapshot。重跑过程中历史 recorded/expert 的部分单次灌水量超过当前 gym-DSSAT action 上限并被裁剪；因此 LC2010 图表反映的是本轮当前环境约束下 DSSAT 实际执行后的基线过程。这个边界只影响 LC2010 的本轮过程图解释，不回写或覆盖旧 027/031 结果。

## 终值摘要

| year | scenario | final_grain_kg_ha | irrigation_event_total_mm | nitrogen_event_total_kg_ha | wp_et_kg_m3 | pfp_n_kg_kg | max_water_stress_wspd | max_nitrogen_stress_nstd | common_reward_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | null | 3193.0 | 0.0 | 0.0 | 1.21 |  | 0.0 | 0.49 | 0.0 |
| 2005 | recorded_farmer | 11769.0 | 150.0 | 300.0 | 3.37 | 39.2 | 0.0 | 0.012 | 6926.0 |
| 2005 | dssat_auto | 3193.0 | 0.0 | 0.0 | 1.21 |  | 0.0 | 0.49 | 0.0 |
| 2005 | official_extension_expert | 12084.0 | 228.8 | 247.0 | 3.43 | 48.8 | 0.0 | 0.012 | 7427.2 |
| 2005 | rl_candidate | 11675.0 | 75.0 | 200.0 | 3.28 | 58.4 | 0.0 | 0.171 | 7407.0 |
| 2006 | null | 2982.0 | 0.0 | 0.0 | 1.19 |  | 0.0 | 0.504 | 0.0 |
| 2006 | recorded_farmer | 9190.0 | 150.0 | 300.0 | 3.01 | 30.6 | 0.0 | 0.012 | 4558.0 |
| 2006 | dssat_auto | 2982.0 | 0.0 | 0.0 | 1.19 |  | 0.0 | 0.504 | 0.0 |
| 2006 | official_extension_expert | 9445.0 | 198.8 | 247.0 | 3.11 | 38.2 | 0.0 | 0.012 | 5029.2 |
| 2006 | rl_candidate | 9180.0 | 75.0 | 200.0 | 2.99 | 45.9 | 0.0 | 0.012 | 5123.0 |
| 2007 | null | 2804.0 | 0.0 | 0.0 | 1.62 |  | 0.0 | 0.455 | 0.0 |
| 2007 | recorded_farmer | 8048.0 | 150.0 | 300.0 | 2.89 | 26.8 | 0.0 | 0.012 | 3594.0 |
| 2007 | dssat_auto | 2804.0 | 0.0 | 0.0 | 1.62 |  | 0.0 | 0.455 | 0.0 |
| 2007 | official_extension_expert | 8106.0 | 198.8 | 247.0 | 2.85 | 32.8 | 0.0 | 0.012 | 3868.2 |
| 2007 | rl_candidate | 8101.0 | 75.0 | 200.0 | 3.01 | 40.5 | 0.0 | 0.012 | 4222.0 |
| 2008 | null | 3081.0 | 0.0 | 0.0 | 1.12 |  | 0.0 | 0.48 | 0.0 |
| 2008 | recorded_farmer | 10834.0 | 150.0 | 300.0 | 3.17 | 36.1 | 0.0 | 0.012 | 6103.0 |
| 2008 | dssat_auto | 3081.0 | 0.0 | 0.0 | 1.12 |  | 0.0 | 0.48 | 0.0 |
| 2008 | official_extension_expert | 10864.0 | 198.8 | 247.0 | 3.1 | 43.9 | 0.0 | 0.024 | 6349.2 |
| 2008 | rl_candidate | 10769.0 | 75.0 | 200.0 | 3.07 | 53.8 | 0.134 | 0.121 | 6613.0 |
| 2009 | null | 2992.0 | 0.0 | 0.0 | 1.09 |  | 0.0 | 0.48 | 0.0 |
| 2009 | recorded_farmer | 9265.0 | 150.0 | 300.0 | 2.95 | 30.9 | 0.0 | 0.012 | 4623.0 |
| 2009 | dssat_auto | 2992.0 | 0.0 | 0.0 | 1.09 |  | 0.0 | 0.48 | 0.0 |
| 2009 | official_extension_expert | 9270.0 | 198.8 | 247.0 | 2.9 | 37.5 | 0.0 | 0.012 | 4844.2 |
| 2009 | rl_candidate | 9288.0 | 75.0 | 200.0 | 2.82 | 46.4 | 0.0 | 0.012 | 5221.0 |
| 2010 | null | 2776.0 | 0.0 | 0.0 | 1.25 |  | 0.0 | 0.476 | 0.0 |
| 2010 | recorded_farmer | 8341.0 | 150.0 | 300.0 | 2.88 | 27.8 | 0.0 | 0.012 | 3915.0 |
| 2010 | dssat_auto | 2776.0 | 0.0 | 0.0 | 1.25 |  | 0.0 | 0.476 | 0.0 |
| 2010 | official_extension_expert | 8349.0 | 198.8 | 247.0 | 2.87 | 33.7 | 0.0 | 0.012 | 4139.2 |
| 2010 | rl_candidate | 8349.0 | 75.0 | 200.0 | 2.8 | 41.7 | 0.085 | 0.012 | 4498.0 |
| 2011 | null | 2861.0 | 0.0 | 0.0 | 1.07 |  | 0.0 | 0.484 | 0.0 |
| 2011 | recorded_farmer | 9125.0 | 150.0 | 300.0 | 2.82 | 30.4 | 0.0 | 0.015 | 4614.0 |
| 2011 | dssat_auto | 2861.0 | 0.0 | 0.0 | 1.07 |  | 0.0 | 0.484 | 0.0 |
| 2011 | official_extension_expert | 9104.0 | 198.8 | 247.0 | 2.82 | 36.8 | 0.0 | 0.023 | 4809.2 |
| 2011 | rl_candidate | 9050.0 | 75.0 | 200.0 | 2.79 | 45.2 | 0.0 | 0.023 | 5114.0 |
| 2012 | null | 2936.0 | 0.0 | 0.0 | 1.23 |  | 0.0 | 0.473 | 0.0 |
| 2012 | recorded_farmer | 8986.0 | 150.0 | 300.0 | 2.63 | 30.0 | 0.0 | 0.012 | 4400.0 |
| 2012 | dssat_auto | 2936.0 | 0.0 | 0.0 | 1.23 |  | 0.0 | 0.473 | 0.0 |
| 2012 | official_extension_expert | 9018.0 | 228.8 | 247.0 | 2.58 | 36.4 | 0.0 | 0.012 | 4618.2 |
| 2012 | rl_candidate | 8996.0 | 75.0 | 200.0 | 2.72 | 45.0 | 0.0 | 0.012 | 4985.0 |
| 2013 | null | 2971.0 | 0.0 | 0.0 | 1.08 |  | 0.0 | 0.494 | 0.0 |
| 2013 | recorded_farmer | 8702.0 | 150.0 | 300.0 | 2.55 | 29.0 | 0.023 | 0.012 | 4081.0 |
| 2013 | dssat_auto | 2971.0 | 0.0 | 0.0 | 1.08 |  | 0.0 | 0.494 | 0.0 |
| 2013 | official_extension_expert | 8723.0 | 198.8 | 247.0 | 2.61 | 35.2 | 0.0 | 0.012 | 4318.2 |
| 2013 | rl_candidate | 8766.0 | 75.0 | 200.0 | 2.54 | 43.8 | 0.081 | 0.012 | 4720.0 |
| 2014 | null | 3146.0 | 0.0 | 0.0 | 1.26 |  | 0.0 | 0.486 | 0.0 |
| 2014 | recorded_farmer | 10517.0 | 150.0 | 300.0 | 3.05 | 35.1 | 0.0 | 0.012 | 5721.0 |
| 2014 | dssat_auto | 3103.0 | 42.1 | 0.0 | 1.24 |  | 0.0 | 0.486 | -42.1 |
| 2014 | official_extension_expert | 10522.0 | 198.8 | 247.0 | 3.09 | 42.5 | 0.0 | 0.024 | 5942.2 |
| 2014 | rl_candidate | 10488.0 | 75.0 | 200.0 | 2.98 | 52.4 | 0.0 | 0.088 | 6267.0 |
| 2015 | null | 3063.0 | 0.0 | 0.0 | 1.31 |  | 0.0 | 0.485 | 0.0 |
| 2015 | recorded_farmer | 10052.0 | 150.0 | 300.0 | 3.16 | 33.5 | 0.0 | 0.012 | 5339.0 |
| 2015 | dssat_auto | 3063.0 | 0.0 | 0.0 | 1.31 |  | 0.0 | 0.485 | 0.0 |
| 2015 | official_extension_expert | 10371.0 | 198.8 | 247.0 | 3.25 | 41.9 | 0.0 | 0.012 | 5874.2 |
| 2015 | rl_candidate | 10113.0 | 75.0 | 200.0 | 3.13 | 50.6 | 0.0 | 0.012 | 5975.0 |
| 2016 | null | 2875.0 | 0.0 | 0.0 | 1.17 |  | 0.0 | 0.461 | 0.0 |
| 2016 | recorded_farmer | 8364.0 | 150.0 | 300.0 | 2.59 | 27.9 | 0.0 | 0.025 | 3839.0 |
| 2016 | dssat_auto | 2875.0 | 0.0 | 0.0 | 1.17 |  | 0.0 | 0.461 | 0.0 |
| 2016 | official_extension_expert | 8565.0 | 198.8 | 247.0 | 2.63 | 34.6 | 0.0 | 0.012 | 4256.2 |
| 2016 | rl_candidate | 8389.0 | 75.0 | 200.0 | 2.57 | 41.9 | 0.0 | 0.012 | 4439.0 |
| 2017 | null | 2914.0 | 0.0 | 0.0 | 1.22 |  | 0.0 | 0.488 | 0.0 |
| 2017 | recorded_farmer | 9041.0 | 150.0 | 300.0 | 2.72 | 30.1 | 0.0 | 0.012 | 4477.0 |
| 2017 | dssat_auto | 2916.0 | 42.4 | 0.0 | 1.19 |  | 0.0 | 0.488 | -40.4 |
| 2017 | official_extension_expert | 9064.0 | 198.8 | 247.0 | 2.81 | 36.6 | 0.0 | 0.012 | 4716.2 |
| 2017 | rl_candidate | 9048.0 | 75.0 | 200.0 | 2.73 | 45.2 | 0.0 | 0.012 | 5059.0 |
| 2018 | null | 2676.0 | 0.0 | 0.0 | 1.21 |  | 0.0 | 0.477 | 0.0 |
| 2018 | recorded_farmer | 7913.0 | 150.0 | 300.0 | 2.57 | 26.4 | 0.0 | 0.012 | 3587.0 |
| 2018 | dssat_auto | 2714.0 | 42.4 | 0.0 | 1.16 |  | 0.0 | 0.477 | -4.4 |
| 2018 | official_extension_expert | 7917.0 | 198.8 | 247.0 | 2.56 | 32.0 | 0.0 | 0.012 | 3807.2 |
| 2018 | rl_candidate | 8233.0 | 75.0 | 200.0 | 2.61 | 41.2 | 0.106 | 0.012 | 4482.0 |
| 2019 | null | 2982.0 | 0.0 | 0.0 | 1.23 |  | 0.0 | 0.504 | 0.0 |
| 2019 | recorded_farmer | 9805.0 | 150.0 | 300.0 | 2.81 | 32.7 | 0.0 | 0.012 | 5173.0 |
| 2019 | dssat_auto | 2982.0 | 41.7 | 0.0 | 1.23 |  | 0.0 | 0.504 | -41.7 |
| 2019 | official_extension_expert | 9726.0 | 198.8 | 247.0 | 2.73 | 39.3 | 0.0 | 0.046 | 5310.2 |
| 2019 | rl_candidate | 9807.0 | 75.0 | 200.0 | 2.75 | 49.0 | 0.0 | 0.117 | 5750.0 |
| 2020 | null | 2476.0 | 0.0 | 0.0 | 1.05 |  | 0.0 | 0.49 | 0.0 |
| 2020 | recorded_farmer | 7762.0 | 150.0 | 300.0 | 2.31 | 25.9 | 0.0 | 0.012 | 3636.0 |
| 2020 | dssat_auto | 2476.0 | 0.0 | 0.0 | 1.05 |  | 0.0 | 0.49 | 0.0 |
| 2020 | official_extension_expert | 7775.0 | 198.8 | 247.0 | 2.29 | 31.4 | 0.0 | 0.012 | 3865.2 |
| 2020 | rl_candidate | 7751.0 | 75.0 | 200.0 | 2.33 | 38.8 | 0.0 | 0.012 | 4200.0 |

## QA 摘要

| check | total | passed |
| --- | --- | --- |
| daily_irrigation_matches_summary_event_total | 80 | 80 |
| daily_irrigation_sum_matches_summary | 80 | 80 |
| daily_nitrogen_matches_summary_event_total | 80 | 80 |
| daily_nitrogen_sum_matches_summary | 80 | 80 |
| five_scenario_rain_total_invariant | 16 | 16 |
| no_duplicate_scenario_date_dap | 16 | 16 |
| rain_not_all_zero_unless_weather_is_zero | 80 | 80 |
| snapshot_exists | 80 | 80 |
| temperature_source_anomaly_count_documented | 80 | 80 |

## QA 未通过项

无记录。

## 图件索引

| year | png | svg |
| --- | --- | --- |
| 2005 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2005_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2005_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2006 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2006_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2006_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2007 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2007_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2007_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2008 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2008_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2008_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2009 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2009_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2009_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2010 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2010_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2010_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2011 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2011_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2011_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2012 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2012_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2012_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2013 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2013_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2013_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2014 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2014_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2014_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2015 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2015_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2015_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2016 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2016_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2016_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2017 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2017_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2017_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2018 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2018_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2018_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2019 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2019_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2019_75k_ppo_five_scenario_daily_snapshot_derived.svg |
| 2020 | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2020_75k_ppo_five_scenario_daily_snapshot_derived.png | benchmark_results/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild/figures/032_20_lc2020_75k_ppo_five_scenario_daily_snapshot_derived.svg |

## 解释边界

- 如果某个年份 null 情景没有 WSPD，并不自动说明图错；DSSAT 的 WSPD 是由土壤水、根系、蒸散需求等共同决定，不等同于降雨少。
- 本轮修复的是日值来源一致性。若重建图里仍出现 PPO 独有胁迫，需要再做动作删除/降档反事实审计，不能仅凭图直接下因果结论。
