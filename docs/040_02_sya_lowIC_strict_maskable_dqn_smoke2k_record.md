# 040_02 SYA lowIC 严格 MaskableDQN 记录

## 结论先说

- 本任务使用项目本地 StrictMaskableDQN；mask 进入探索、贪心、replay 和 Bellman target。
- 除 DQN 框架外，输入、年份、动作空间、约束和奖励均沿用 040_00/040_01。

## 固定配置

- 输入根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：`SYA`
- seed：`0`
- 训练步数：`2000`
- checkpoint：`1000, 2000`

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

| algorithm | station_code | site | train_years | seed | checkpoint_step | saved_step | run_status | model_path | model_sha256 | optimizer_updates | target_updates | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StrictMaskableDQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 1000 | 1000 | ok_existing | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | 408f32f8677ddbe99862b298fd7c5ada28a72bfc706bd894bb246d2edf8835e2 | 0 | 8 | Existing checkpoint reused. |
| StrictMaskableDQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 2000 | 2000 | ok_existing | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | 5e3b943eb3229fd42a73d7312d08f81dd3034760d4415997585569b75ab70d8c | 244 | 16 | Existing checkpoint reused. |

## 训练年份采样次数

无记录。

## 更新日志节选

无记录。

## 验证集 checkpoint 汇总

| station_code | site | checkpoint_step | validation_years | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_PFP_N | mean_swfac_stress_days_gt_0p05 | mean_nstres_days_gt_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 1000 | 10 | 7718.0295 | 150.0 | 240.0 | 32.1585 | 11.1 | 0.0 |
| SYA | SY | 2000 | 10 | 8111.5738 | 150.0 | 240.0 | 33.7982 | 9.9 | 0.4 |

## 逐年验证结果

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | site | year | seed | checkpoint_step | saved_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 144 | 5518.8458 | 150.0 | 240.0 | 4974.6458 | 22.9952 | 0.328 | 0.2534 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 27 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2014_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 6054.6295 | 150.0 | 240.0 | 5510.4295 | 25.2276 | 0.4126 | 0.128 | 60.0 | 80.0 | False | 5 | 6 | 2 | 1 | 11 | 0 | DAP1 I0/N40; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0; DAP30 I0/N40; DAP37 I0/N40 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2015_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 7993.3002 | 150.0 | 240.0 | 7449.1002 | 33.3054 | 0.7193 | 0.5924 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2016_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 130 | 4697.6178 | 150.0 | 240.0 | 4153.4178 | 19.5734 | 0.1987 | 0.6448 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 31 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2017_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 129 | 7031.5613 | 150.0 | 240.0 | 6487.3613 | 29.2982 | 0.5668 | 0.0 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 9 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2018_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 140 | 8482.5983 | 150.0 | 240.0 | 7938.3983 | 35.3442 | 0.7968 | 0.7506 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 8 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2019_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 6205.6232 | 150.0 | 240.0 | 5661.4232 | 25.8568 | 0.4371 | 0.827 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 21 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2020_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 10372.3401 | 150.0 | 240.0 | 9828.1401 | 43.2181 | 1.0956 | 0.9948 | 60.0 | 80.0 | False | 5 | 6 | 2 | 1 | 0 | 0 | DAP1 I0/N40; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0; DAP30 I0/N40; DAP37 I0/N40 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2021_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 142 | 10898.7341 | 150.0 | 240.0 | 10354.5341 | 45.4114 | 1.1783 | 0.4509 | 60.0 | 160.0 | False | 5 | 4 | 2 | 1 | 0 | 0 | DAP1 I0/N120; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2022_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 9925.0446 | 150.0 | 240.0 | 9380.8446 | 41.3544 | 1.0256 | 1.6196 | 60.0 | 80.0 | False | 5 | 6 | 2 | 1 | 4 | 0 | DAP1 I0/N40; DAP2 I30/N0; DAP8 I30/N0; DAP9 I0/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP29 I30/N0; DAP30 I0/N40; DAP37 I0/N40 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt1000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2023_ckpt1000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 1000 | 1000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 148 | 5958.8593 | 150.0 | 240.0 | 5414.6593 | 24.8286 | 0.3973 | 0.0 | 45.0 | 120.0 | False | 4 | 2 | 5 | 1 | 25 | 0 | DAP1 I0/N120; DAP5 I45/N0; DAP12 I15/N120; DAP19 I45/N0; DAP26 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2014_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 142 | 8089.9841 | 150.0 | 240.0 | 7545.7841 | 33.7083 | 0.734 | 0.0 | 45.0 | 120.0 | False | 4 | 2 | 5 | 1 | 10 | 0 | DAP1 I0/N120; DAP5 I45/N0; DAP12 I15/N120; DAP19 I45/N0; DAP26 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2015_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 7995.0574 | 150.0 | 240.0 | 7450.8574 | 33.3127 | 0.72 | 1.0056 | 15.0 | 160.0 | False | 4 | 3 | 8 | 1 | 0 | 0 | DAP1 I0/N120; DAP8 I15/N40; DAP15 I45/N80; DAP22 I45/N0; DAP29 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2016_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 132 | 5380.7843 | 150.0 | 240.0 | 4836.5843 | 22.4199 | 0.306 | 0.0 | 15.0 | 160.0 | False | 4 | 3 | 8 | 1 | 25 | 0 | DAP1 I0/N120; DAP8 I15/N40; DAP15 I45/N80; DAP22 I45/N0; DAP29 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2017_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 128 | 7558.667 | 150.0 | 240.0 | 7014.467 | 31.4944 | 0.6513 | 1.2367 | 15.0 | 160.0 | False | 4 | 3 | 8 | 1 | 10 | 4 | DAP1 I0/N120; DAP8 I15/N40; DAP15 I45/N80; DAP22 I45/N0; DAP29 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2018_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 140 | 8496.405 | 150.0 | 240.0 | 7952.205 | 35.4017 | 0.7982 | 0.0 | 60.0 | 240.0 | False | 4 | 2 | 3 | 1 | 8 | 0 | DAP1 I0/N120; DAP3 I45/N0; DAP10 I15/N120; DAP17 I45/N0; DAP24 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2019_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 6656.7059 | 150.0 | 240.0 | 6112.5059 | 27.7363 | 0.5088 | 1.2759 | 15.0 | 160.0 | False | 4 | 3 | 8 | 1 | 18 | 0 | DAP1 I0/N120; DAP8 I15/N40; DAP15 I45/N80; DAP22 I45/N0; DAP29 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2020_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 10432.478 | 150.0 | 240.0 | 9888.278 | 43.4687 | 1.1041 | 0.0 | 60.0 | 240.0 | False | 4 | 2 | 1 | 1 | 0 | 0 | DAP1 I0/N120; DAP1 I45/N0; DAP8 I15/N120; DAP15 I45/N0; DAP22 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2021_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 10415.3064 | 150.0 | 240.0 | 9871.1064 | 43.3971 | 1.1604 | 59.0065 | 15.0 | 160.0 | False | 4 | 3 | 8 | 1 | 2 | 0 | DAP1 I0/N120; DAP8 I15/N40; DAP15 I45/N80; DAP22 I45/N0; DAP29 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2022_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 2000 | 2000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 10131.4905 | 150.0 | 240.0 | 9587.2905 | 42.2145 | 1.0566 | 0.0 | 15.0 | 160.0 | False | 4 | 3 | 8 | 1 | 1 | 0 | DAP1 I0/N120; DAP8 I15/N40; DAP15 I45/N80; DAP22 I45/N0; DAP29 I45/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt2000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn_smoke2k/daily_outputs/SYA/2023_ckpt2000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 2000 | 2000 |
