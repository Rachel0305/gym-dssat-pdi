# 018_02 官方农技推广 extension expert baseline 小试记录

## 本轮原则

- 不修改 DQN 奖励函数。
- 不重新训练 DQN。
- 不改动原始输入文件。
- 只新增 `extension_expert_fixed_dap` 管理情景。
- HLA2010 使用推文表1，YC2014 使用推文表3。
- 本轮采用固定 DAP 映射，不使用实测生育期，也不使用 DSSAT 事后生育期对齐。

## Extension expert 结果

| station | year | scenario | final_gwad | final_cwad | rain_total | action_irrigation_total | action_fertilizer_total | event_irrigation_total | event_fertilizer_total | max_water_stress | max_nitrogen_stress | final_dap | run_dir | label | irrigation_total | fertilizer_total | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | 2010.00 | extension_expert_fixed_dap | 7854.00 | 20571.00 | 0.00 | 266.25 | 300.00 | 266.10 | 300.00 | 0.42 | 0.02 | 130.00 | DSSAT_auto_validation/extension_expert_baseline_018_02/HLA2010/extension_expert_fixed_dap | Extension expert fixed DAP | 266.10 | 300.00 | 018_02 extension expert fixed DAP |
| YC | 2014.00 | extension_expert_fixed_dap | 9417.00 | 20481.00 | 0.00 | 228.75 | 247.50 | 228.80 | 247.00 | 0.00 | 0.01 | 104.00 | DSSAT_auto_validation/extension_expert_baseline_018_02/YC2014/extension_expert_fixed_dap | Extension expert fixed DAP | 228.80 | 247.00 | 018_02 extension expert fixed DAP |

## 与现有结果合并后的初步对照

| station | year | label | final_gwad | final_cwad | irrigation_total | fertilizer_total | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | 2010.00 | DQN best checkpoint | 7853.67 | 20886.03 | 120.00 | 0.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| HLA | 2010.00 | DSSAT auto | 7854.00 | 20874.00 | 190.40 | 0.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| HLA | 2010.00 | Recorded expert | 7679.00 | 20665.00 | 30.00 | 165.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| HLA | 2010.00 | Null | 6956.00 | 19344.00 | 0.00 | 0.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| HLA | 2010.00 | DQN best checkpoint | 7573.02 | 20255.35 | 60.00 | 0.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| HLA | 2010.00 | DSSAT auto | 7854.00 | 20874.00 | 190.40 | 0.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| HLA | 2010.00 | Recorded expert | 7679.00 | 20665.00 | 30.00 | 165.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| HLA | 2010.00 | Null | 6956.00 | 19344.00 | 0.00 | 0.00 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_2015_four_scenario_final_dqn_all_summary.csv |
| YC | 2014.00 | dqn_baseline_relative_9action | 9418.00 | 20514.00 | 120.00 | 300.00 | DSSAT_auto_validation/yc2014_baseline_relative_dqn_015_10/seed0/dqn_baseline_relative_9action/015_10_yc2014_baseline_relative_summary.csv |
| HLA | 2010.00 | Extension expert fixed DAP | 7854.00 | 20571.00 | 266.10 | 300.00 | 018_02 extension expert fixed DAP |
| YC | 2014.00 | Extension expert fixed DAP | 9417.00 | 20481.00 | 228.80 | 247.00 | 018_02 extension expert fixed DAP |

## 解释

本轮只检验新增官方农技推广 expert baseline 对现有叙事的冲击。若 extension expert 产量更高且资源投入也更高，应解释为“高投入推广方案”；DQN 若接近该产量但用水氮更少，则可作为节水节氮策略优势。

## 输出文件

- schedule: `DSSAT_auto_validation/extension_expert_baseline_018_02/extension_expert_schedule.csv`
- summary: `DSSAT_auto_validation/extension_expert_baseline_018_02/018_02_extension_expert_summary.csv`
- combined: `DSSAT_auto_validation/extension_expert_baseline_018_02/018_02_extension_expert_combined_comparison.csv`
- figure: `DSSAT_auto_validation/extension_expert_baseline_018_02/figures/018_02_extension_expert_summary.png`