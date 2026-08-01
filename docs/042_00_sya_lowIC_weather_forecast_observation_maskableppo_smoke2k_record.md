# 042_00_smoke2k SYA lowIC 天气预报增强 observation MaskablePPO 记录

## 结论边界

- 本任务改变 agent observation 信息结构：新增当天降雨、当天 Tmin、近 7 天降雨、未来 7 天降雨、未来 7 天平均温度。
- 未来天气使用历史天气构造，属于 perfect weather forecast 场景。
- reward、动作档位和安全约束沿用 040_40；本任务不是 reward 调参。
- 训练步数：`2000`；checkpoint：`1000, 2000`。

## 输入路径和数据安全

- lowIC 输入目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- lowIC 输入目录存在：`True`
- 039_00 authoritative lowIC 审计状态：`{'exists': True, 'issue_count': 0, 'status': 'pass', 'note': 'empty issue file means no authoritative issues'}`

## observation smoke audit

| station_code | year | base_observation_dim | enhanced_observation_dim | obs_len | legal_action_count | feature_rain_today_mm | feature_tmin_today_c | feature_rain_past7_mm | feature_rain_future7_mm | feature_tmean_future7_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | 2005 | 25 | 30 | 30 | 9 | 0.0 | 4.4753 | 24.7 | 8.7 | 10.8174 |
| SYA | 2014 | 25 | 30 | 30 | 9 | 0.0 | 3.8 | 0.0 | 0.0 | 17.2357 |
| SYA | 2023 | 25 | 30 | 30 | 9 | 0.0 | 4.4 | 0.0 | 16.0 | 9.5143 |

## 年份划分

| station_code | year | split |
| --- | --- | --- |
| SYA | 2005 | train |
| SYA | 2006 | train |
| SYA | 2007 | train |
| SYA | 2008 | train |
| SYA | 2009 | train |
| SYA | 2010 | train |
| SYA | 2011 | train |
| SYA | 2012 | train |
| SYA | 2013 | train |
| SYA | 2014 | validation |
| SYA | 2015 | validation |
| SYA | 2016 | validation |
| SYA | 2017 | validation |
| SYA | 2018 | validation |
| SYA | 2019 | validation |
| SYA | 2020 | validation |
| SYA | 2021 | validation |
| SYA | 2022 | validation |
| SYA | 2023 | validation |

## 训练 checkpoint 库存

| station_code | site | train_years | seed | checkpoint_step | run_status | model_path | model_sha256 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 1000 | ok | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | 7e94fefdaca75eaa33b04227a9cd4f28cb41cb7f905586c73c5bf904fda9ba93 |  |
| SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 2000 | ok | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | a0b6b1ed814da6a6d5f4865d6b1b5857b48fbd6fb16549556b5e8f40b2cf38ea |  |

## 验证集按 checkpoint 汇总

| station_code | site | checkpoint_step | validation_years | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_PFP_N | max_nstres | max_swfac | mean_early_reserve_penalty_unscaled | any_metric_win_four_count | mean_gap_yield_vs_four_max | mean_gap_pfp_n_vs_four_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 1000 | 10 | 5752.0721 | 66.0 | 200.0 | 28.7604 | 0.2135 | 0.9999 |  | 0 |  |  |
| SYA | SY | 2000 | 10 | 4390.5295 | 45.0 | 84.0 | 53.4247 | 0.6405 | 1.0 |  | 0 |  |  |

## 逐年验证结果

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | year | seed | checkpoint_step | max_swfac | swfac_days_gt_0p001 | swfac_days_gt_0p01 | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p001 | nstres_days_gt_0p01 | baseline_rows | any_metric_win_four |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 144 | 3656.5948 | 75.0 | 200.0 | 3258.0948 | 18.282974243164062 | -2.3983 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 39 | 0 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2014_seed0_ckpt1000_daily.csv | SYA | 2014 | 0 | 1000 | 0.9984 | 39 | 39 | 39 | 0.0122 | 9 | 2 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 141 | 2847.9446 | 75.0 | 200.0 | 2449.4446 | 14.239723205566406 | -1.8795 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 21 | 0 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2015_seed0_ckpt1000_daily.csv | SYA | 2015 | 0 | 1000 | 0.9776 | 21 | 21 | 21 | 0.0122 | 5 | 2 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 141 | 6715.3845 | 75.0 | 200.0 | 6316.8845 | 33.57692260742188 | 0.4679 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 5 | 0 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2016_seed0_ckpt1000_daily.csv | SYA | 2016 | 0 | 1000 | 0.7606 | 5 | 5 | 5 | 0.0152 | 8 | 4 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 130 | 3448.6087 | 75.0 | 200.0 | 3050.1087 | 17.243043518066408 | -2.4768 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 42 | 0 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2017_seed0_ckpt1000_daily.csv | SYA | 2017 | 0 | 1000 | 0.9999 | 43 | 42 | 42 | 0.0144 | 7 | 3 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 129 | 5558.3752 | 75.0 | 200.0 | 5159.8752 | 27.791876220703124 | -0.4296 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 16 | 0 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2018_seed0_ckpt1000_daily.csv | SYA | 2018 | 0 | 1000 | 0.9412 | 16 | 16 | 16 | 0.017 | 8 | 4 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 140 | 5837.5244 | 75.0 | 200.0 | 5439.0244 | 29.1876220703125 | -0.8304 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 21 | 0 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2019_seed0_ckpt1000_daily.csv | SYA | 2019 | 0 | 1000 | 0.9292 | 23 | 22 | 21 | 0.013 | 6 | 3 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 138 | 2891.041 | 30.0 | 200.0 | 2542.041 | 14.45520477294922 | -2.33 | 0.0 | 30.0 | 200.0 | False | 1 | 2 | 8 | 1.0 | 36 | 0 | DAP1 I0/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2020_seed0_ckpt1000_daily.csv | SYA | 2020 | 0 | 1000 | 0.9647 | 36 | 36 | 36 | 0.0354 | 9 | 5 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 138 | 9244.8663 | 75.0 | 200.0 | 8846.3663 | 46.22433166503906 | 0.8816 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 5 | 0 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2021_seed0_ckpt1000_daily.csv | SYA | 2021 | 0 | 1000 | 0.7718 | 5 | 5 | 5 | 0.0136 | 7 | 3 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 142 | 11046.8958 | 75.0 | 200.0 | 10648.3958 | 55.23447875976562 | 1.4122 | 65.25 | 75.0 | 200.0 | False | 2 | 2 | 1 | 1.0 | 0 | 6 | DAP1 I45/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2022_seed0_ckpt1000_daily.csv | SYA | 2022 | 0 | 1000 | 0.0 | 0 | 0 | 0 | 0.2135 | 14 | 10 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 138 | 6273.4857 | 30.0 | 200.0 | 5924.4857 | 31.367428588867188 | -0.9531 | 0.0 | 30.0 | 200.0 | False | 1 | 2 | 8 | 1.0 | 25 | 0 | DAP1 I0/N120; DAP8 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt1000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2023_seed0_ckpt1000_daily.csv | SYA | 2023 | 0 | 1000 | 0.9872 | 25 | 25 | 25 | 0.0161 | 6 | 3 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 144 | 3452.9526 | 45.0 | 80.0 | 3277.0526 | 43.16190719604492 | -2.3998 | 0.0 | 45.0 | 80.0 | False | 1 | 1 | 2 | 2.0 | 43 | 15 | DAP2 I45/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2014_seed0_ckpt2000_daily.csv | SYA | 2014 | 0 | 2000 | 1.0 | 43 | 43 | 43 | 0.1303 | 39 | 30 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 141 | 2436.185 | 45.0 | 80.0 | 2260.285 | 30.452312469482422 | -1.9284 | 0.0 | 45.0 | 80.0 | False | 1 | 1 | 2 | 2.0 | 21 | 20 | DAP2 I45/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2015_seed0_ckpt2000_daily.csv | SYA | 2015 | 0 | 2000 | 1.0 | 21 | 21 | 21 | 0.0957 | 37 | 31 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 141 | 5863.6902 | 45.0 | 80.0 | 5687.7902 | 73.29612731933594 | 0.2259 | 0.0 | 45.0 | 80.0 | False | 1 | 1 | 2 | 2.0 | 9 | 48 | DAP2 I45/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2016_seed0_ckpt2000_daily.csv | SYA | 2016 | 0 | 2000 | 0.8859 | 9 | 9 | 9 | 0.5974 | 68 | 64 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 130 | 1763.9508 | 30.0 | 0.0 | 1730.9508 |  | -2.4683 | 43.5 | 30.0 | 0.0 | False | 1 | 0 | 1 |  | 41 | 61 | DAP1 I30/N0 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2017_seed0_ckpt2000_daily.csv | SYA | 2017 | 0 | 2000 | 0.9415 | 42 | 41 | 41 | 0.3798 | 69 | 65 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 129 | 5355.8246 | 75.0 | 160.0 | 5020.5246 | 33.47390365600586 | -0.52 | 1.1409 | 45.0 | 80.0 | False | 2 | 2 | 2 | 2.0 | 16 | 0 | DAP2 I45/N80; DAP11 I30/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2018_seed0_ckpt2000_daily.csv | SYA | 2018 | 0 | 2000 | 0.9686 | 16 | 16 | 16 | 0.017 | 8 | 4 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 140 | 1238.7813 | 30.0 | 0.0 | 1205.7813 |  | -1.8064 | 43.5 | 30.0 | 0.0 | False | 1 | 0 | 1 |  | 19 | 91 | DAP1 I30/N0 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2019_seed0_ckpt2000_daily.csv | SYA | 2019 | 0 | 2000 | 0.898 | 19 | 19 | 19 | 0.5344 | 99 | 95 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 138 | 3316.2656 | 45.0 | 80.0 | 3140.3656 | 41.45331954956055 | -1.9255 | 0.0 | 45.0 | 80.0 | False | 1 | 1 | 3 | 3.0 | 35 | 0 | DAP3 I45/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2020_seed0_ckpt2000_daily.csv | SYA | 2020 | 0 | 2000 | 0.998 | 35 | 35 | 35 | 0.0413 | 9 | 6 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 138 | 5931.6534 | 45.0 | 80.0 | 5755.7534 | 74.14566802978516 | -0.2036 | 0.0 | 45.0 | 80.0 | False | 1 | 1 | 2 | 2.0 | 10 | 28 | DAP2 I45/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2021_seed0_ckpt2000_daily.csv | SYA | 2021 | 0 | 2000 | 0.8942 | 10 | 10 | 10 | 0.6311 | 46 | 35 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 142 | 7824.5471 | 45.0 | 80.0 | 7648.6471 | 97.80683898925781 | 0.6054 | 0.0 | 45.0 | 80.0 | False | 1 | 1 | 2 | 2.0 | 0 | 35 | DAP2 I45/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2022_seed0_ckpt2000_daily.csv | SYA | 2022 | 0 | 2000 | 0.0 | 0 | 0 | 0 | 0.6405 | 53 | 44 | 0 |  |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 138 | 6721.4441 | 45.0 | 200.0 | 6355.9441 | 33.60722045898437 | -0.7074 | 0.0 | 45.0 | 200.0 | False | 1 | 2 | 8 | 1.0 | 25 | 0 | DAP1 I0/N120; DAP8 I45/N80 | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip | benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo_smoke2k/daily_outputs/SYA/SYA_2023_seed0_ckpt2000_daily.csv | SYA | 2023 | 0 | 2000 | 0.9513 | 25 | 25 | 25 | 0.0161 | 6 | 3 | 0 |  |
