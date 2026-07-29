# 040_01 SYA lowIC 自由时序 DQN 对照记录

## 结论先说

- 本任务是 040_00 的 DQN 对照；只换算法，不改 lowIC 输入、年份划分、动作约束或奖励。
- 训练使用 SB3 DQN；评估使用 Q 值 masked-greedy，以避免确定性评估阶段选择非法动作。
- 注意：SB3 DQN 训练采样阶段并非原生 MaskableDQN，非法动作仍由环境安全层兜底。

## 固定配置

- 输入根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：`SYA`
- seed：`0`
- 训练步数：`100000`
- checkpoint：`25000, 50000, 75000, 100000`

## SYA lowIC 可用性分类

| station_code | year | lowIC_usability_class | lowIC_usability_reason |
| --- | --- | --- | --- |
| SYA | 2005 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2006 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2007 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2008 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2009 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2010 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2011 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2012 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2013 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2014 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2015 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2016 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2017 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2018 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2019 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2020 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2021 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2022 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |
| SYA | 2023 | A_RL_training_candidate | lowIC creates null stress/yield contrast and official expert remains buffered |

## 年份划分

| station_code | year | split | selected_for_train | selected_for_eval |
| --- | --- | --- | --- | --- |
| SYA | 2005 | train | True | True |
| SYA | 2006 | train | True | True |
| SYA | 2007 | train | True | True |
| SYA | 2008 | train | True | True |
| SYA | 2009 | train | True | True |
| SYA | 2010 | train | True | True |
| SYA | 2011 | train | True | True |
| SYA | 2012 | train | True | True |
| SYA | 2013 | train | True | True |
| SYA | 2014 | validation | False | True |
| SYA | 2015 | validation | False | True |
| SYA | 2016 | validation | False | True |
| SYA | 2017 | validation | False | True |
| SYA | 2018 | validation | False | True |
| SYA | 2019 | validation | False | True |
| SYA | 2020 | validation | False | True |
| SYA | 2021 | validation | False | True |
| SYA | 2022 | validation | False | True |
| SYA | 2023 | validation | False | True |

## 训练 checkpoint 库存

| algorithm | station_code | site | train_years | seed | checkpoint_step | run_status | model_path | model_sha256 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 25000 | ok | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | 09ec6ee2e8670708802f078a38eea97397b9d421c329053f30ba46726d05e707 |  |
| DQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 50000 | ok | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | 1c4baf3ba082837d4a5182651a9922a357731d6ff34d7d23479ea264df2b3a23 |  |
| DQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 75000 | ok | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | eae2817410ac9d98241d0ffa0a282551913c0a53579b3686296597d36f1befd7 |  |
| DQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 100000 | ok | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | f28cec5ec96abb2e7f54c4ed833e6b29853de1189c89b9f7cc8b8df80431aac3 |  |

## 训练年份采样次数

| station_code | year | episode_count |
| --- | --- | --- |
| SYA | 2005 | 78 |
| SYA | 2006 | 64 |
| SYA | 2007 | 80 |
| SYA | 2008 | 70 |
| SYA | 2009 | 70 |
| SYA | 2010 | 85 |
| SYA | 2011 | 84 |
| SYA | 2012 | 80 |
| SYA | 2013 | 91 |

## 验证集 checkpoint 汇总

| station_code | site | checkpoint_step | validation_years | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_PFP_N | mean_swfac_stress_days_gt_0p05 | mean_nstres_days_gt_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 25000 | 10 | 7193.8406 | 150.0 | 240.0 | 29.9743 | 13.8 | 0.0 |
| SYA | SY | 50000 | 10 | 7718.0955 | 150.0 | 240.0 | 32.1587 | 11.1 | 0.0 |
| SYA | SY | 75000 | 10 | 7721.742 | 150.0 | 240.0 | 32.1739 | 11.0 | 0.0 |
| SYA | SY | 100000 | 10 | 7439.8965 | 150.0 | 240.0 | 30.9996 | 12.1 | 0.0 |

## 逐年验证结果

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | site | year | seed | checkpoint_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQN | dqn_stress_aware_5k_seed0 | ok | 144 | 4798.4244 | 150.0 | 240.0 | 4254.2244 | 19.9934 | 0.214 | 0.0 | 30.0 | 240.0 | False | 9 | 2 | 2 | 1 | 30 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2014_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 4671.8793 | 150.0 | 240.0 | 4127.6793 | 19.4662 | 0.1943 | 0.3839 | 30.0 | 240.0 | False | 9 | 2 | 2 | 1 | 15 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2015_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 7960.2161 | 150.0 | 240.0 | 7416.0161 | 33.1676 | 0.7135 | 0.0 | 30.0 | 240.0 | False | 9 | 2 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2016_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 130 | 3872.1973 | 150.0 | 240.0 | 3327.9973 | 16.1342 | 0.0685 | 0.8946 | 30.0 | 240.0 | False | 10 | 2 | 2 | 1 | 38 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I15/N0; DAP84 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2017_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 129 | 6945.8484 | 150.0 | 240.0 | 6401.6484 | 28.941 | 0.5532 | 0.0 | 30.0 | 240.0 | False | 9 | 2 | 2 | 1 | 11 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2018_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 140 | 8580.3088 | 150.0 | 240.0 | 8036.1088 | 35.7513 | 0.8122 | 0.7506 | 30.0 | 200.0 | False | 9 | 3 | 2 | 1 | 7 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2019_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 5180.9363 | 150.0 | 240.0 | 4636.7363 | 21.5872 | 0.2744 | 0.0 | 30.0 | 240.0 | False | 10 | 2 | 2 | 1 | 24 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I15/N0; DAP75 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2020_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 10391.9958 | 150.0 | 240.0 | 9847.7958 | 43.3 | 1.0977 | 0.0 | 30.0 | 240.0 | False | 10 | 2 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I15/N0; DAP79 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2021_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 142 | 11074.8083 | 150.0 | 240.0 | 10530.6083 | 46.145 | 1.2282 | 22.5877 | 30.0 | 240.0 | False | 10 | 2 | 2 | 1 | 1 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N120; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I15/N0; DAP80 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2022_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 8461.7914 | 150.0 | 240.0 | 7917.5914 | 35.2575 | 0.7933 | 0.5628 | 30.0 | 200.0 | False | 9 | 3 | 2 | 1 | 12 | 0 | DAP1 I0/N120; DAP2 I15/N0; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0; DAP57 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt25000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2023_ckpt25000_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 25000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 144 | 5518.8458 | 150.0 | 240.0 | 4974.6458 | 22.9952 | 0.328 | 0.2534 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 27 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2014_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 6056.4899 | 150.0 | 240.0 | 5512.2899 | 25.2354 | 0.4129 | 0.128 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 11 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2015_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 7993.3002 | 150.0 | 240.0 | 7449.1002 | 33.3054 | 0.7193 | 0.5924 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2016_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 130 | 4697.6178 | 150.0 | 240.0 | 4153.4178 | 19.5734 | 0.1987 | 0.6448 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 31 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2017_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 129 | 7031.5613 | 150.0 | 240.0 | 6487.3613 | 29.2982 | 0.5668 | 0.0 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 9 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2018_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 140 | 8482.5983 | 150.0 | 240.0 | 7938.3983 | 35.3442 | 0.7968 | 0.7506 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 8 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2019_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 6205.6232 | 150.0 | 240.0 | 5661.4232 | 25.8568 | 0.4371 | 0.827 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 21 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2020_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 10375.2014 | 150.0 | 240.0 | 9831.0014 | 43.23 | 1.0955 | 0.4654 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2021_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 142 | 10898.7341 | 150.0 | 240.0 | 10354.5341 | 45.4114 | 1.1783 | 0.4509 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2022_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 9920.9827 | 150.0 | 240.0 | 9376.7827 | 41.3374 | 1.0239 | 0.5628 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 4 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt50000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2023_ckpt50000_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 50000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 144 | 5530.1038 | 150.0 | 240.0 | 4985.9038 | 23.0421 | 0.3305 | 0.9052 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 27 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2014_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 6070.9058 | 150.0 | 240.0 | 5526.7058 | 25.2954 | 0.415 | 0.0 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 11 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2015_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 8003.0005 | 150.0 | 240.0 | 7458.8005 | 33.3458 | 0.7213 | 1.0056 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2016_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 130 | 4626.7654 | 150.0 | 240.0 | 4082.5654 | 19.2782 | 0.1876 | 0.8059 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 31 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2017_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 129 | 7019.4128 | 150.0 | 240.0 | 6475.2128 | 29.2476 | 0.5649 | 0.0 | 75.0 | 240.0 | False | 5 | 2 | 2 | 1 | 9 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N120; DAP15 I45/N0; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2018_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 140 | 8506.1584 | 150.0 | 240.0 | 7961.9584 | 35.4423 | 0.8006 | 0.7904 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 8 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2019_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 6211.5955 | 150.0 | 240.0 | 5667.3955 | 25.8816 | 0.4385 | 1.2759 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 20 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2020_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 10393.5486 | 150.0 | 240.0 | 9849.3486 | 43.3065 | 1.098 | 0.0 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2021_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 142 | 10927.1228 | 150.0 | 240.0 | 10382.9228 | 45.5297 | 1.1835 | 1.2179 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2022_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 9928.8068 | 150.0 | 240.0 | 9384.6068 | 41.37 | 1.0246 | 0.0 | 75.0 | 160.0 | False | 5 | 3 | 2 | 1 | 4 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I45/N40; DAP15 I45/N80; DAP22 I15/N0; DAP29 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt75000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2023_ckpt75000_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 75000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 144 | 5291.2476 | 150.0 | 240.0 | 4747.0476 | 22.0469 | 0.2921 | 0.2534 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 27 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2014_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 4851.4001 | 150.0 | 240.0 | 4307.2001 | 20.2142 | 0.2224 | 0.128 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 13 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2015_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 141 | 7983.3899 | 150.0 | 240.0 | 7439.1899 | 33.2641 | 0.7178 | 0.5924 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2016_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 130 | 4057.7231 | 150.0 | 240.0 | 3513.5231 | 16.9072 | 0.0976 | 0.6448 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 37 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2017_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 129 | 7004.9921 | 150.0 | 240.0 | 6460.7921 | 29.1875 | 0.5626 | 0.0 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 9 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2018_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 140 | 8469.2279 | 150.0 | 240.0 | 7925.0279 | 35.2884 | 0.7947 | 0.7506 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 8 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2019_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 6225.0299 | 150.0 | 240.0 | 5680.8299 | 25.9376 | 0.4402 | 0.827 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 20 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2020_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 10359.5837 | 150.0 | 240.0 | 9815.3837 | 43.1649 | 1.0931 | 0.4654 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2021_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 142 | 10892.5122 | 150.0 | 240.0 | 10348.3122 | 45.3855 | 1.1773 | 0.4509 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2022_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 100000 |
| DQN | dqn_stress_aware_5k_seed0 | ok | 138 | 9263.858 | 150.0 | 240.0 | 8719.658 | 38.5994 | 0.9201 | 0.5628 | 90.0 | 160.0 | False | 6 | 4 | 2 | 1 | 7 | 0 | DAP1 I0/N120; DAP2 I45/N0; DAP8 I45/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0 | benchmark_results/040_01_sya_lowIC_free_timing_dqn/models/SYA/SYA_lowIC_free_timing_dqn_seed0_ckpt100000.zip | benchmark_results/040_01_sya_lowIC_free_timing_dqn/daily_outputs/SYA/2023_ckpt100000_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 100000 |

## 解释边界

- 本任务不是 DQN 参数搜索。
- 如果 DQN 表现弱于 PPO，只能说明当前 DQN 配置在这个 lowIC/SYA half-split 设置下不占优。
- 如果 DQN 表现优于 PPO，下一步仍需跨 seed 或站点复核，不能直接宣布 DQN 全面优于 PPO。
