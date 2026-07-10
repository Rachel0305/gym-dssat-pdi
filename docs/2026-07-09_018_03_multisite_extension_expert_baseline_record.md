# 018_03 五站点官方农技推广 expert baseline 扩展记录

## 本轮原则

- 只新增 `official extension expert fixed DAP` 情景。
- 不修改 DQN 奖励函数。
- 不训练 DQN。
- 不改原始输入文件。
- HLA/SY 使用 PDF 表 1；YC/FQ/LC 使用 PDF 表 3。
- 采用固定 DAP 映射，不用事后模拟生育期。

## Extension expert replay 结果

| site | station | year | status | final_gwad | final_cwad | event_irrigation_total | event_fertilizer_total | max_water_stress | max_nitrogen_stress | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | Hailun | 2010 | ok | 7854.000 | 20571.000 | 266.100 | 300.000 | 0.421 | 0.016 |  |
| SY | Shenyang | 2014 | ok | 11077.000 | 19522.000 | 266.100 | 300.000 | 0.000 | 0.012 |  |
| YC | Yucheng | 2014 | ok | 9417.000 | 20481.000 | 228.800 | 247.000 | 0.000 | 0.013 |  |
| FQ | Fengqiu | 2016 | ok | 7940.000 | 13769.000 | 198.800 | 247.000 | 0.000 | 0.012 |  |
| LC | Luancheng | 2010 | ok | 8739.000 | 16376.000 | 198.800 | 247.000 | 0.000 | 0.019 |  |

## 与已有情景的干净合并表

| site | station | year | scenario_label | grain_yield_kg_ha | biomass_kg_ha | irrigation_mm | nitrogen_kg_ha | max_water_stress | max_nitrogen_stress | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | Fengqiu | 2016 | Null | 7066.000 | 13148.000 | 0.000 | 0.000 | 0.657 | 0.012 | DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv |
| FQ | Fengqiu | 2016 | Recorded/farmer practice | 7933.000 | 14008.000 | 75.000 | 144.000 | 0.373 | 0.012 | DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv |
| FQ | Fengqiu | 2016 | DSSAT auto | 8012.000 | 14095.000 | 59.900 | 0.000 | 0.000 | 0.012 | DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv |
| FQ | Fengqiu | 2016 | DQN best checkpoint | 7995.000 | 14078.000 | 60.000 | 0.000 | 0.050 | 0.012 | DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv |
| FQ | Fengqiu | 2016 | Official extension expert fixed DAP | 7940.000 | 13769.000 | 198.800 | 247.000 | 0.000 | 0.012 | 018_03_extension_expert_summary.csv |
| HLA | Hailun | 2010 | Null | 6956.000 | 19344.000 | 0.000 | 0.000 | 0.919 | 0.157 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_four_scenario_final_dqn_seed0_summary.csv |
| HLA | Hailun | 2010 | Recorded expert | 7679.000 | 20665.000 | 30.000 | 165.000 | 0.811 | 0.016 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_four_scenario_final_dqn_seed0_summary.csv |
| HLA | Hailun | 2010 | DSSAT auto | 7854.000 | 20874.000 | 190.400 | 0.000 | 0.416 | 0.035 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_four_scenario_final_dqn_seed0_summary.csv |
| HLA | Hailun | 2010 | DQN best checkpoint | 7853.665 | 20886.030 | 120.000 | 0.000 | 0.416 | 0.062 | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_four_scenario_final_dqn_seed0_summary.csv |
| HLA | Hailun | 2010 | Official extension expert fixed DAP | 7854.000 | 20571.000 | 266.100 | 300.000 | 0.421 | 0.016 | 018_03_extension_expert_summary.csv |
| LC | Luancheng | 2010 | Null | 8051.000 |  | 0.000 | 0.000 |  |  | run_lc2010_baseline_relative_dqn_smoke_017_12.py constants |
| LC | Luancheng | 2010 | Recorded/farmer practice | 8732.000 |  |  |  |  |  | run_lc2010_baseline_relative_dqn_smoke_017_12.py constants |
| LC | Luancheng | 2010 | DSSAT auto | 8738.000 |  |  |  |  |  | run_lc2010_baseline_relative_dqn_smoke_017_12.py constants |
| LC | Luancheng | 2010 | Official extension expert fixed DAP | 8739.000 | 16376.000 | 198.800 | 247.000 | 0.000 | 0.019 | 018_03_extension_expert_summary.csv |
| SY | Shenyang | 2014 | Null | 2769.000 | 5648.000 | 0.000 | 0.000 | 0.000 | 0.442 | DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_summary.csv |
| SY | Shenyang | 2014 | Recorded expert | 9593.000 | 18302.000 | 0.000 | 293.000 | 0.807 | 0.012 | DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_summary.csv |
| SY | Shenyang | 2014 | DSSAT auto | 2724.000 | 5715.000 | 33.400 | 0.000 | 0.000 | 0.442 | DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_summary.csv |
| SY | Shenyang | 2014 | DQN ckpt15000 | 11216.000 | 20250.000 | 120.000 | 300.000 | 0.000 | 0.012 | DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_summary.csv |
| SY | Shenyang | 2014 | Official extension expert fixed DAP | 11077.000 | 19522.000 | 266.100 | 300.000 | 0.000 | 0.012 | 018_03_extension_expert_summary.csv |
| YC | Yucheng | 2014 | Null | 7825.000 | 17996.000 | 0.000 | 0.000 | 0.922 | 0.381 | DSSAT_auto_validation/yc2014_formal_four_scenario_015_06/seed0_seed1_best/015_06_yc2014_formal_four_scenario_summary.csv |
| YC | Yucheng | 2014 | Recorded/farmer practice | 9418.000 | 20514.000 | 120.000 | 374.000 | 0.000 | 0.013 | DSSAT_auto_validation/yc2014_formal_four_scenario_015_06/seed0_seed1_best/015_06_yc2014_formal_four_scenario_summary.csv |
| YC | Yucheng | 2014 | DSSAT auto | 8713.000 | 18945.000 | 86.500 | 0.000 | 0.000 | 0.436 | DSSAT_auto_validation/yc2014_formal_four_scenario_015_06/seed0_seed1_best/015_06_yc2014_formal_four_scenario_summary.csv |
| YC | Yucheng | 2014 | DQN best checkpoint | 9418.000 | 20513.000 | 120.000 | 250.000 | 0.000 | 0.013 | DSSAT_auto_validation/yc2014_formal_four_scenario_015_06/seed0_seed1_best/015_06_yc2014_formal_four_scenario_summary.csv |
| YC | Yucheng | 2014 | Official extension expert fixed DAP | 9417.000 | 20481.000 | 228.800 | 247.000 | 0.000 | 0.013 | 018_03_extension_expert_summary.csv |

## 初步解释

这一步的目的不是证明 DQN 最优，而是把导师要求的官方推广 expert baseline 加入当前叙事。汇报时应区分：

1. `Recorded/farmer practice`：历史记录/农民管理。
2. `Official extension expert fixed DAP`：根据官方农技推广方案中值换算得到的固定 DAP 管理。
3. `DSSAT auto`：DSSAT 原生自动管理。
4. `DQN best checkpoint`：现有 DQN 结果，不在本轮重新训练。

如果 DQN 与官方推广 expert 产量相近但投入更低，可解释为“用更低资源逼近高投入推广方案的产量平台”；如果 DQN 产量低但更省资源，则需由导师决定是否接受产量-资源权衡；如果 DQN 产量和效率均低于官方推广方案，则后续需要讨论奖励函数和约束设置。

## 输出文件

- schedule: `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_extension_expert_schedule.csv`
- summary: `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_extension_expert_summary.csv`
- daily: `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_extension_expert_daily.csv`
- events: `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_extension_expert_events.csv`
- clean comparison: `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv`
- figure: `DSSAT_auto_validation/extension_expert_baseline_018_03/figures/018_03_extension_expert_summary.png`