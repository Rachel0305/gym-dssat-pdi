# 043_01_sya_lowIC_04215_policy_input_sensitivity_audit

## 结论先说

- 分支：`C_weak_or_no_policy_input_sensitivity_detected`。
- 本任务没有训练、没有调参、没有修改 reward、没有修改 DSSAT 输入。
- 审计对象是 042_15 冻结口径中的 042_11 ckpt25k binary-timing MaskablePPO。
- 这一步检查的是：冻结 PPO 在同一 action mask 下，对已有 observation 变量的局部扰动是否改变动作概率或确定性动作。

## 为什么这一步能支撑“完美天气预报”的论文叙事

- 如果当前 observation 中没有 rain/tmin/未来7天降雨等信息，PPO 不可能直接对这些变量作出响应。
- 如果已有的 tmax/srad/soil/stress 扰动也只带来弱响应，说明现有输入结构更容易学成阶段/预算模板。
- 因此，后续加入历史天气构造的完美天气预报窗口，可被表述为改变决策信息结构，而不是事后调参。

## 固定对象

- 模型：`benchmark_results/042_11_sya_lowIC_binary_timing_training_length_curve/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip`
- 输入：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 年份：`2014–2023`
- DAP：`[1, 2, 31, 38, 51, 61, 91]`

## 当前 policy observation 中不可直接响应的天气变量

| variable | in_policy_observation | interpretation |
| --- | --- | --- |
| rain | False | not directly visible to current 042_15 policy |
| tmin | False | not directly visible to current 042_15 policy |
| rain_past7 | False | not directly visible to current 042_15 policy |
| rain_future7 | False | not directly visible to current 042_15 policy |
| tmean_future7 | False | not directly visible to current 042_15 policy |
| tmax_future7 | False | not directly visible to current 042_15 policy |
| srad_future7 | False | not directly visible to current 042_15 policy |

## observation 变量清单

| index | name | lower_name | is_soil_water | is_swfac | is_nstres | is_tmax | is_srad | is_rain | is_tmin | name_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cumsumfert | cumsumfert | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 1 | dap | dap | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 2 | dtt | dtt | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 3 | ep | ep | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 4 | grnwt | grnwt | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 5 | istage | istage | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 6 | nstres | nstres | False | False | True | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 7 | rtdep | rtdep | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 8 | srad | srad | False | False | False | False | True | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 9 | sw_1 | sw_1 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 10 | sw_2 | sw_2 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 11 | sw_3 | sw_3 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 12 | sw_4 | sw_4 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 13 | sw_5 | sw_5 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 14 | sw_6 | sw_6 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 15 | sw_7 | sw_7 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 16 | sw_8 | sw_8 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 17 | sw_9 | sw_9 | True | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 18 | swfac | swfac | False | True | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 19 | tmax | tmax | False | False | False | True | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 20 | topwt | topwt | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 21 | totir | totir | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 22 | vstage | vstage | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 23 | wtdep | wtdep | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |
| 24 | xlai | xlai | False | False | False | False | False | False | False | env_chain[0].observation_variables;expanded_sw_to_9_layers |

## 抽样状态上下文

| state_id | year | dap | step_index | baseline_action | action_label | action_irrigation_mm | action_n_kg_ha | valid_action_count | mask_true_indices | policy_obs_tmax | policy_obs_srad | policy_obs_swfac | policy_obs_nstres | policy_obs_grnwt | policy_obs_topwt | policy_obs_totir | policy_obs_dap | context_rain | context_tmin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014_dap1_step0 | 2014 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 23.6 | 21.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2014_dap2_step1 | 2014 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 19.0 | 8.7 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2014_dap1_step5 | 2014 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 21.8 | 20.2 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2014_dap2_step6 | 2014 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 25.1 | 20.4 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2014_dap31_step35 | 2014 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 28.7 | 28.0 | 0.0 | 0.0 | 0.0 | 42.0822 | 45.0 | 31.0 |  |  |
| 2014_dap38_step42 | 2014 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 29.4 | 29.4 | 0.0 | 0.0 | 0.0 | 120.5176 | 45.0 | 38.0 |  |  |
| 2014_dap51_step55 | 2014 | 51 | 55 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 26.0 | 25.9 | 0.0 | 0.0 | 0.0 | 650.1342 | 45.0 | 51.0 |  |  |
| 2014_dap61_step65 | 2014 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 27.1 | 11.4 | 0.0 | 0.0 | 0.0 | 1456.4186 | 135.0 | 61.0 |  |  |
| 2014_dap91_step95 | 2014 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 32.4 | 19.7 | 0.0 | 0.0 | 0.0 | 6816.1089 | 180.0 | 91.0 |  |  |
| 2015_dap1_step0 | 2015 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 16.7 | 16.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2015_dap2_step1 | 2015 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 20.8 | 21.0 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2015_dap1_step5 | 2015 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 20.0 | 13.5 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2015_dap2_step6 | 2015 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 23.0 | 23.3 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2015_dap31_step35 | 2015 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 30.5 | 28.7 | 0.0 | 0.0 | 0.0 | 61.9979 | 45.0 | 31.0 |  |  |
| 2015_dap38_step42 | 2015 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 29.6 | 23.8 | 0.0 | 0.0 | 0.0 | 196.8526 | 45.0 | 38.0 |  |  |
| 2015_dap51_step55 | 2015 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 25.0 | 14.1 | 0.0 | 0.0 | 0.0 | 730.5945 | 90.0 | 51.0 |  |  |
| 2015_dap61_step65 | 2015 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 29.4 | 24.5 | 0.0 | 0.0 | 0.0 | 1300.0181 | 135.0 | 61.0 |  |  |
| 2015_dap91_step95 | 2015 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 30.7 | 23.5 | 0.0 | 0.0 | 0.0 | 6584.168 | 180.0 | 91.0 |  |  |
| 2016_dap1_step0 | 2016 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 22.1 | 14.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2016_dap2_step1 | 2016 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 17.5 | 6.8 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2016_dap1_step5 | 2016 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 21.9 | 22.0 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2016_dap2_step6 | 2016 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 24.4 | 22.8 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2016_dap31_step35 | 2016 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 29.5 | 27.6 | 0.0 | 0.0 | 0.0 | 44.9437 | 45.0 | 31.0 |  |  |
| 2016_dap38_step42 | 2016 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 29.2 | 27.8 | 0.0 | 0.0 | 0.0 | 114.5666 | 45.0 | 38.0 |  |  |
| 2016_dap51_step55 | 2016 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 32.0 | 24.9 | 0.0 | 0.0 | 0.0 | 591.0253 | 45.0 | 51.0 |  |  |
| 2016_dap61_step65 | 2016 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 28.4 | 26.7 | 0.0 | 0.0 | 0.0 | 1261.9648 | 135.0 | 61.0 |  |  |
| 2016_dap91_step95 | 2016 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 31.1 | 23.9 | 0.0 | 0.0 | 0.0 | 6426.832 | 180.0 | 91.0 |  |  |
| 2017_dap1_step0 | 2017 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 23.9 | 15.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2017_dap2_step1 | 2017 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 22.3 | 13.2 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2017_dap1_step5 | 2017 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 20.6 | 23.6 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2017_dap2_step6 | 2017 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 24.9 | 23.9 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2017_dap31_step35 | 2017 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 28.0 | 26.4 | 0.0 | 0.0 | 0.0 | 83.6028 | 45.0 | 31.0 |  |  |
| 2017_dap38_step42 | 2017 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 28.9 | 21.2 | 0.0 | 0.0 | 0.0 | 182.9313 | 45.0 | 38.0 |  |  |
| 2017_dap51_step55 | 2017 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 26.8 | 30.3 | 0.7834 | 0.0 | 0.0 | 404.2142 | 45.0 | 51.0 |  |  |
| 2017_dap61_step65 | 2017 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 26.9 | 18.3 | 0.9234 | 0.0 | 0.0 | 484.5413 | 45.0 | 61.0 |  |  |
| 2017_dap91_step95 | 2017 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 30.9 | 12.0 | 0.0 | 0.0 | 312.7265 | 6621.2524 | 180.0 | 91.0 |  |  |
| 2018_dap1_step0 | 2018 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 21.5 | 24.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2018_dap2_step1 | 2018 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 23.2 | 21.5 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2018_dap1_step5 | 2018 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 20.3 | 11.2 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2018_dap2_step6 | 2018 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 11.4 | 2.4 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2018_dap31_step35 | 2018 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 27.7 | 27.0 | 0.0 | 0.0 | 0.0 | 91.411 | 45.0 | 31.0 |  |  |
| 2018_dap38_step42 | 2018 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 22.9 | 4.4 | 0.0 | 0.0 | 0.0 | 207.6525 | 45.0 | 38.0 |  |  |
| 2018_dap51_step55 | 2018 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 22.9 | 6.3 | 0.0 | 0.0 | 0.0 | 848.758 | 90.0 | 51.0 |  |  |
| 2018_dap61_step65 | 2018 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 27.1 | 29.3 | 0.0 | 0.0 | 0.0 | 1460.0146 | 135.0 | 61.0 |  |  |
| 2018_dap91_step95 | 2018 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 32.0 | 24.4 | 0.0 | 0.0 | 0.0 | 7852.6543 | 180.0 | 91.0 |  |  |
| 2019_dap1_step0 | 2019 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 22.2 | 21.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2019_dap2_step1 | 2019 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 21.8 | 17.6 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2019_dap1_step5 | 2019 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 19.8 | 24.9 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2019_dap2_step6 | 2019 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 28.3 | 19.0 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2019_dap31_step35 | 2019 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 24.0 | 26.6 | 0.0 | 0.0 | 0.0 | 69.7977 | 45.0 | 31.0 |  |  |
| 2019_dap38_step42 | 2019 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 24.5 | 28.4 | 0.0 | 0.0 | 0.0 | 195.9117 | 45.0 | 38.0 |  |  |
| 2019_dap51_step55 | 2019 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 25.5 | 22.1 | 0.0 | 0.0 | 0.0 | 727.8979 | 90.0 | 51.0 |  |  |
| 2019_dap61_step65 | 2019 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 29.5 | 21.1 | 0.0 | 0.0 | 0.0 | 1437.5251 | 135.0 | 61.0 |  |  |
| 2019_dap91_step95 | 2019 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 32.5 | 24.7 | 0.0 | 0.0 | 0.0 | 7362.7734 | 180.0 | 91.0 |  |  |
| 2020_dap1_step0 | 2020 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 25.4 | 17.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2020_dap2_step1 | 2020 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 21.4 | 17.2 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2020_dap1_step5 | 2020 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 11.0 | 9.7 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2020_dap2_step6 | 2020 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 6.4 | 16.4 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2020_dap31_step35 | 2020 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 21.6 | 23.8 | 0.0 | 0.0 | 0.0 | 40.0451 | 45.0 | 31.0 |  |  |
| 2020_dap38_step42 | 2020 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 19.3 | 5.2 | 0.0 | 0.0 | 0.0 | 70.7751 | 45.0 | 38.0 |  |  |
| 2020_dap51_step55 | 2020 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 29.6 | 18.9 | 0.0 | 0.0 | 0.0 | 403.3912 | 45.0 | 51.0 |  |  |
| 2020_dap61_step65 | 2020 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 31.4 | 28.3 | 0.0 | 0.0 | 0.0 | 1213.0394 | 90.0 | 61.0 |  |  |
| 2020_dap91_step95 | 2020 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 25.6 | 3.0 | 0.0 | 0.0 | 0.0 | 6939.2437 | 180.0 | 91.0 |  |  |
| 2021_dap1_step0 | 2021 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 14.1 | 15.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2021_dap2_step1 | 2021 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 9.8 | 6.5 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2021_dap1_step5 | 2021 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 28.5 | 23.0 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2021_dap2_step6 | 2021 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 22.2 | 7.8 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2021_dap31_step35 | 2021 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 28.4 | 25.1 | 0.0 | 0.0 | 0.0 | 46.2639 | 45.0 | 31.0 |  |  |
| 2021_dap38_step42 | 2021 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 20.1 | 14.1 | 0.0 | 0.0 | 0.0 | 92.0992 | 45.0 | 38.0 |  |  |
| 2021_dap51_step55 | 2021 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 25.0 | 5.7 | 0.0 | 0.0 | 0.0 | 391.7488 | 45.0 | 51.0 |  |  |
| 2021_dap61_step65 | 2021 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 26.3 | 26.5 | 0.0 | 0.0 | 0.0 | 1035.9602 | 90.0 | 61.0 |  |  |
| 2021_dap91_step95 | 2021 | 91 | 95 | 2 | I45_N0 | 45.0 | 0.0 | 2 | 0,2 | 31.5 | 22.3 | 0.0 | 0.0 | 0.0 | 6494.3564 | 180.0 | 91.0 |  |  |
| 2022_dap1_step0 | 2022 | 1 | 0 | 2 | I45_N0 | 45.0 | 0.0 | 4 | 0,1,2,3 | 20.9 | 24.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |
| 2022_dap2_step1 | 2022 | 2 | 1 | 1 | I0_N80 | 0.0 | 80.0 | 2 | 0,1 | 22.6 | 23.8 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |  |  |
| 2022_dap1_step5 | 2022 | 1 | 5 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 23.6 | 18.3 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 1.0 |  |  |
| 2022_dap2_step6 | 2022 | 2 | 6 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 18.5 | 26.2 | 0.0 | 0.0 | 0.0 | 0.0 | 45.0 | 2.0 |  |  |
| 2022_dap31_step35 | 2022 | 31 | 35 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 30.2 | 23.0 | 0.0 | 0.0 | 0.0 | 50.9109 | 45.0 | 31.0 |  |  |
| 2022_dap38_step42 | 2022 | 38 | 42 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 30.4 | 24.1 | 0.0 | 0.0 | 0.0 | 144.5441 | 45.0 | 38.0 |  |  |
| 2022_dap51_step55 | 2022 | 51 | 55 | 0 | I0_N0 | 0.0 | 0.0 | 4 | 0,1,2,3 | 24.6 | 15.1 | 0.0 | 0.0 | 0.0 | 464.8463 | 45.0 | 51.0 |  |  |
| 2022_dap61_step65 | 2022 | 61 | 65 | 0 | I0_N0 | 0.0 | 0.0 | 1 | 0 | 31.6 | 28.5 | 0.0 | 0.0 | 0.0 | 1138.2618 | 90.0 | 61.0 |  |  |

## 按扰动类型汇总

| scenario | n_states | n_changed | change_rate | mean_prob_l1 | mean_prob_kl | n_irrigation_action | n_n_action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dry_soil | 90 | 0 | 0.0 | 0.0007 | 0.0 | 21 | 11 |
| high_nstres | 90 | 0 | 0.0 | 0.0005 | 0.0 | 21 | 11 |
| high_srad | 90 | 0 | 0.0 | 0.0128 | 0.0004 | 21 | 11 |
| high_swfac | 90 | 0 | 0.0 | 0.0005 | 0.0 | 21 | 11 |
| hot_tmax | 90 | 0 | 0.0 | 0.0152 | 0.0006 | 21 | 11 |
| low_srad | 90 | 0 | 0.0 | 0.0122 | 0.0004 | 21 | 11 |
| wet_soil | 90 | 0 | 0.0 | 0.0016 | 0.0 | 21 | 11 |

## 按 DAP 和扰动类型汇总

| dap | scenario | n_states | n_changed | change_rate | mean_prob_l1 |
| --- | --- | --- | --- | --- | --- |
| 1 | dry_soil | 20 | 0 | 0.0 | 0.0 |
| 1 | high_nstres | 20 | 0 | 0.0 | 0.0005 |
| 1 | high_srad | 20 | 0 | 0.0 | 0.0142 |
| 1 | high_swfac | 20 | 0 | 0.0 | 0.001 |
| 1 | hot_tmax | 20 | 0 | 0.0 | 0.0216 |
| 1 | low_srad | 20 | 0 | 0.0 | 0.0169 |
| 1 | wet_soil | 20 | 0 | 0.0 | 0.0018 |
| 2 | dry_soil | 20 | 0 | 0.0 | 0.0013 |
| 2 | high_nstres | 20 | 0 | 0.0 | 0.001 |
| 2 | high_srad | 20 | 0 | 0.0 | 0.0199 |
| 2 | high_swfac | 20 | 0 | 0.0 | 0.0005 |
| 2 | hot_tmax | 20 | 0 | 0.0 | 0.0243 |
| 2 | low_srad | 20 | 0 | 0.0 | 0.02 |
| 2 | wet_soil | 20 | 0 | 0.0 | 0.0039 |
| 31 | dry_soil | 10 | 0 | 0.0 | 0.0013 |
| 31 | high_nstres | 10 | 0 | 0.0 | 0.0002 |
| 31 | high_srad | 10 | 0 | 0.0 | 0.0211 |
| 31 | high_swfac | 10 | 0 | 0.0 | 0.0008 |
| 31 | hot_tmax | 10 | 0 | 0.0 | 0.0115 |
| 31 | low_srad | 10 | 0 | 0.0 | 0.0153 |
| 31 | wet_soil | 10 | 0 | 0.0 | 0.0017 |
| 38 | dry_soil | 10 | 0 | 0.0 | 0.002 |
| 38 | high_nstres | 10 | 0 | 0.0 | 0.0011 |
| 38 | high_srad | 10 | 0 | 0.0 | 0.0211 |
| 38 | high_swfac | 10 | 0 | 0.0 | 0.0008 |
| 38 | hot_tmax | 10 | 0 | 0.0 | 0.0277 |
| 38 | low_srad | 10 | 0 | 0.0 | 0.0188 |
| 38 | wet_soil | 10 | 0 | 0.0 | 0.0013 |
| 51 | dry_soil | 10 | 0 | 0.0 | 0.0 |
| 51 | high_nstres | 10 | 0 | 0.0 | 0.0001 |
| 51 | high_srad | 10 | 0 | 0.0 | 0.0039 |
| 51 | high_swfac | 10 | 0 | 0.0 | 0.0 |
| 51 | hot_tmax | 10 | 0 | 0.0 | 0.0052 |
| 51 | low_srad | 10 | 0 | 0.0 | 0.0013 |
| 51 | wet_soil | 10 | 0 | 0.0 | 0.0001 |
| 61 | dry_soil | 10 | 0 | 0.0 | 0.0002 |
| 61 | high_nstres | 10 | 0 | 0.0 | 0.0 |
| 61 | high_srad | 10 | 0 | 0.0 | 0.0006 |
| 61 | high_swfac | 10 | 0 | 0.0 | 0.0002 |
| 61 | hot_tmax | 10 | 0 | 0.0 | 0.0007 |
| 61 | low_srad | 10 | 0 | 0.0 | 0.0005 |
| 61 | wet_soil | 10 | 0 | 0.0 | 0.0 |
| 91 | dry_soil | 10 | 0 | 0.0 | 0.0 |
| 91 | high_nstres | 10 | 0 | 0.0 | 0.0 |
| 91 | high_srad | 10 | 0 | 0.0 | 0.0 |
| 91 | high_swfac | 10 | 0 | 0.0 | 0.0 |
| 91 | hot_tmax | 10 | 0 | 0.0 | 0.0 |
| 91 | low_srad | 10 | 0 | 0.0 | 0.0 |
| 91 | wet_soil | 10 | 0 | 0.0 | 0.0 |

## 输出文件

- 详细表：`benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_policy_input_sensitivity_detail.csv`
- 扰动汇总：`benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_policy_input_sensitivity_by_scenario.csv`
- DAP 汇总：`benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_policy_input_sensitivity_by_dap.csv`
- observation 变量：`benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_policy_observation_variables.csv`
- 不可见天气变量：`benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_unobservable_weather_variables.csv`

## 边界

- 这是 policy 局部敏感性审计，不等同于 DSSAT 物理反事实。
- 若某扰动导致 action 改变，只说明冻结 PPO 对该输入维度有局部响应；是否提高产量或效率需要后续季节回放验证。
- 若某扰动不改变 action，也不能说明该变量农学上不重要，只能说明当前 checkpoint 没有把它强烈用于动作选择。
