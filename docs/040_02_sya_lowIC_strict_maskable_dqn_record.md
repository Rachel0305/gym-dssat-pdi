# 040_02 SYA lowIC 严格 MaskableDQN 记录

## 结论先说

- 本任务使用项目本地 StrictMaskableDQN；mask 进入探索、贪心、replay 和 Bellman target。
- 除 DQN 框架外，输入、年份、动作空间、约束和奖励均沿用 040_00/040_01。

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

| algorithm | station_code | site | train_years | seed | checkpoint_step | saved_step | run_status | model_path | model_sha256 | optimizer_updates | target_updates | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StrictMaskableDQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 25000 | 25000 | ok | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | a67618c3dd77a3b3f149d772d8e56cae05e7710669ac7ceeb7bffa96ecbaffed | 5994 | 208 |  |
| StrictMaskableDQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 50000 | 50000 | ok | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | f9c7b40339aee3e181e7e61962b4f9e8150d8e56e877d1d71d025bdc0b770785 | 12244 | 416 |  |
| StrictMaskableDQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 75000 | 75000 | ok | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | b1d4f4205a916e1b771b838ad610d9645267565a4b34bf22d64e48e7828c6e69 | 18494 | 625 |  |
| StrictMaskableDQN | SYA | SY | 2005,2006,2007,2008,2009,2010,2011,2012,2013 | 0 | 100000 | 100000 | ok | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | 9551d7bc49d70108aceb3ab613faf7e73d4cf249712433f58bbe34fcc04d0269 | 24744 | 833 |  |

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

## 更新日志节选

| global_step | loss | mean_chosen_q | mean_target | preclip_grad_norm |
| --- | --- | --- | --- | --- |
| 99844 | 1.0008 | -51.4057 | -50.6225 | 758.5317 |
| 99848 | 1.298 | -50.6063 | -50.3369 | 1449.3623 |
| 99852 | 1.2932 | -51.1996 | -51.314 | 1783.1112 |
| 99856 | 1.1002 | -50.3409 | -50.5975 | 1729.0093 |
| 99860 | 1.5914 | -52.0757 | -51.4793 | 1409.8757 |
| 99864 | 1.2202 | -50.5661 | -49.7653 | 506.2508 |
| 99868 | 1.1196 | -50.6334 | -49.5274 | 1606.9495 |
| 99872 | 1.4149 | -52.316 | -50.8422 | 1928.8267 |
| 99876 | 1.2869 | -53.0925 | -51.6664 | 1926.2416 |
| 99880 | 1.1788 | -51.9358 | -50.65 | 1870.8182 |
| 99884 | 0.7824 | -52.0343 | -51.3432 | 945.5071 |
| 99888 | 1.0877 | -51.2632 | -50.8903 | 1390.1821 |
| 99892 | 1.3703 | -51.1395 | -50.9065 | 1554.4021 |
| 99896 | 1.1479 | -49.9609 | -50.1874 | 1667.999 |
| 99900 | 1.1045 | -49.9486 | -49.9952 | 1704.6913 |
| 99904 | 1.1153 | -51.9011 | -51.2056 | 502.8118 |
| 99908 | 1.2921 | -51.0483 | -49.7892 | 1695.2211 |
| 99912 | 1.3195 | -52.5756 | -51.1543 | 1960.5604 |
| 99916 | 1.3321 | -51.5048 | -50.1683 | 2000.406 |
| 99920 | 1.0868 | -52.337 | -51.2378 | 1806.3264 |
| 99924 | 0.7876 | -50.2071 | -49.6147 | 860.5432 |
| 99928 | 0.8456 | -50.4072 | -50.1519 | 1229.1829 |
| 99932 | 1.4883 | -52.3311 | -52.1077 | 1651.2313 |
| 99936 | 1.4396 | -50.5165 | -50.3109 | 1565.3237 |
| 99940 | 1.0671 | -50.4602 | -50.257 | 1613.5203 |
| 99944 | 0.7895 | -50.6744 | -50.3104 | 467.8938 |
| 99948 | 1.0334 | -51.6068 | -50.5073 | 1707.1919 |
| 99952 | 1.4032 | -51.5803 | -50.1775 | 1940.1593 |
| 99956 | 1.3095 | -51.1082 | -49.6123 | 1993.166 |
| 99960 | 1.0071 | -51.9608 | -50.9821 | 1642.5398 |
| 99964 | 0.6363 | -51.3072 | -50.8798 | 734.8968 |
| 99968 | 1.1541 | -51.6245 | -51.4113 | 1441.0607 |
| 99972 | 1.4273 | -50.881 | -50.8778 | 1749.1613 |
| 99976 | 1.0061 | -50.0947 | -50.4205 | 1696.8016 |
| 99980 | 1.0095 | -51.1829 | -51.1452 | 1479.697 |
| 99984 | 1.0758 | -51.3297 | -50.5476 | 497.8499 |
| 99988 | 1.3496 | -51.8498 | -50.4855 | 1699.621 |
| 99992 | 1.6258 | -52.2184 | -50.4933 | 2021.51 |
| 99996 | 1.3376 | -51.5402 | -50.0861 | 1866.3221 |
| 100000 | 1.5274 | -51.2797 | -49.7693 | 1886.5375 |

## 验证集 checkpoint 汇总

| station_code | site | checkpoint_step | validation_years | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_PFP_N | mean_swfac_stress_days_gt_0p05 | mean_nstres_days_gt_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 25000 | 10 | 6427.4069 | 90.0 | 240.0 | 26.7809 | 17.8 | 0.0 |
| SYA | SY | 50000 | 10 | 4924.6024 | 136.5 | 40.0 | 123.1151 | 11.6 | 63.8 |
| SYA | SY | 75000 | 10 | 2102.0729 | 0.0 | 16.0 | 79.0547 | 23.9 | 58.6 |
| SYA | SY | 100000 | 10 | 1527.2972 | 21.0 | 0.0 |  | 22.2 | 63.3 |

## 逐年验证结果

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | site | year | seed | checkpoint_step | saved_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 144 | 3991.8484 | 90.0 | 240.0 | 3513.6484 | 16.6327 | 0.2178 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 38 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2014_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 3324.5505 | 90.0 | 240.0 | 2846.3505 | 13.8523 | 0.1123 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 20 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2015_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 8032.0789 | 90.0 | 240.0 | 7553.8789 | 33.467 | 0.8561 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 0 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2016_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 130 | 3552.7237 | 90.0 | 240.0 | 3074.5237 | 14.803 | 0.1484 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 40 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2017_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 129 | 5955.7867 | 90.0 | 240.0 | 5477.5867 | 24.8158 | 0.5281 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 14 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2018_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 140 | 6043.4875 | 90.0 | 240.0 | 5565.2875 | 25.1812 | 0.5419 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 17 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2019_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 4362.8961 | 90.0 | 240.0 | 3884.6961 | 18.1787 | 0.2764 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 28 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2020_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 10020.6073 | 90.0 | 240.0 | 9542.4073 | 41.7525 | 1.1703 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 3 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2021_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 142 | 11067.5012 | 90.0 | 240.0 | 10589.3012 | 46.1146 | 1.3357 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 0 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2022_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 7922.5885 | 90.0 | 240.0 | 7444.3885 | 33.0108 | 0.8388 | 65.25 | 90.0 | 240.0 | False | 2 | 2 | 1.0 | 1.0 | 18 | 0 | DAP1 I45/N120; DAP8 I45/N120 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt25000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2023_ckpt25000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 25000 | 25000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 144 | 3881.8729 | 135.0 | 40.0 | 3670.1729 | 97.0468 | 0.4669 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 30 | 69 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2014_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 4321.3251 | 135.0 | 40.0 | 4109.6251 | 108.0331 | 0.5363 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 8 | 68 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2015_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 6223.324 | 135.0 | 40.0 | 6011.624 | 155.5831 | 0.8368 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 0 | 63 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2016_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 130 | 3451.2134 | 135.0 | 40.0 | 3239.5134 | 86.2803 | 0.3988 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 30 | 61 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2017_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 129 | 5030.947 | 150.0 | 40.0 | 4802.747 | 125.7737 | 0.6319 | 65.25 | 90.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 7 | 59 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I45/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2018_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 140 | 4068.945 | 135.0 | 40.0 | 3857.245 | 101.7236 | 0.4964 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 12 | 69 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2019_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 3759.8969 | 135.0 | 40.0 | 3548.1969 | 93.9974 | 0.4476 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 23 | 63 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2020_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 6266.2714 | 135.0 | 40.0 | 6054.5714 | 156.6568 | 0.8436 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 0 | 59 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2021_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 142 | 6972.0044 | 135.0 | 40.0 | 6760.3044 | 174.3001 | 0.9551 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 0 | 62 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2022_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 5270.2246 | 135.0 | 40.0 | 5058.5246 | 131.7556 | 0.6862 | 65.25 | 75.0 | 40.0 | False | 4 | 1 | 1.0 | 2.0 | 6 | 65 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I30/N0; DAP15 I30/N0; DAP22 I30/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt50000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2023_ckpt50000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 50000 | 50000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 152 | 2706.8683 | 0.0 | 40.0 | 2643.6683 | 67.6717 | 0.3645 | 0.0 | 0.0 | 0.0 | False | 0 | 1 |  | 34.0 | 42 | 73 | DAP34 I0/N40 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2014_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 146 | 3455.5542 | 0.0 | 40.0 | 3392.3542 | 86.3889 | 0.4828 | 0.0 | 0.0 | 0.0 | False | 0 | 1 |  | 33.0 | 19 | 69 | DAP33 I0/N40 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2015_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 1402.7704 | 0.0 | 0.0 | 1402.7704 |  | 0.2216 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 13 | 85 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2016_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 62 | 0.0 | 0.0 | 40.0 | -63.2 | 0.0 | -0.0632 | 0.0 | 0.0 | 0.0 | False | 0 | 1 |  | 39.0 | 34 | 0 | DAP39 I0/N40 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2017_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 128 | 1539.5903 | 0.0 | 0.0 | 1539.5903 |  | 0.2433 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 27 | 76 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2018_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 26 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 0 | 0 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2019_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 1286.805 | 0.0 | 0.0 | 1286.805 |  | 0.2033 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 45 | 74 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2020_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 1849.967 | 0.0 | 0.0 | 1849.967 |  | 0.2923 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 11 | 82 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2021_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 6486.3324 | 0.0 | 40.0 | 6423.1324 | 162.1583 | 0.9616 | 0.0 | 0.0 | 0.0 | False | 0 | 1 |  | 36.0 | 18 | 65 | DAP36 I0/N40 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2022_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 2292.8418 | 0.0 | 0.0 | 2292.8418 |  | 0.3623 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 30 | 62 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt75000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2023_ckpt75000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 75000 | 75000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 152 | 1607.5714 | 15.0 | 0.0 | 1591.0714 |  | 0.2375 | 0.0 | 0.0 | 0.0 | False | 1 | 0 | 117.0 |  | 25 | 92 | DAP117 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2014_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2014 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 146 | 1432.6886 | 15.0 | 0.0 | 1416.1886 |  | 0.2099 | 0.0 | 0.0 | 0.0 | False | 1 | 0 | 116.0 |  | 19 | 86 | DAP116 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2015_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2015 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 1402.7704 | 15.0 | 0.0 | 1386.2704 |  | 0.2051 | 0.0 | 0.0 | 0.0 | False | 1 | 0 | 118.0 |  | 13 | 85 | DAP118 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2016_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2016 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 62 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 34 | 0 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2017_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2017 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 128 | 1539.5903 | 30.0 | 0.0 | 1506.5903 |  | 0.2103 | 0.0 | 0.0 | 0.0 | False | 2 | 0 | 108.0 |  | 27 | 76 | DAP108 I15/N0; DAP115 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2018_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2018 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 26 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 | 0.0 | 0.0 | 0.0 | False | 0 | 0 |  |  | 0 | 0 |  | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2019_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2019 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 1334.6432 | 45.0 | 0.0 | 1285.1432 |  | 0.1614 | 0.0 | 0.0 | 0.0 | False | 3 | 0 | 106.0 |  | 45 | 69 | DAP106 I15/N0; DAP113 I15/N0; DAP120 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2020_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2020 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 1887.8139 | 30.0 | 0.0 | 1854.8139 |  | 0.2653 | 0.0 | 0.0 | 0.0 | False | 2 | 0 | 110.0 |  | 11 | 82 | DAP110 I15/N0; DAP117 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2021_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2021 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 141 | 3768.782 | 30.0 | 0.0 | 3735.782 |  | 0.5625 | 0.0 | 0.0 | 0.0 | False | 2 | 0 | 107.0 |  | 18 | 81 | DAP107 I15/N0; DAP114 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2022_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2022 | 0 | 100000 | 100000 |
| StrictMaskableDQN | strictmaskabledqn_stress_aware_5k_seed0 | ok | 138 | 2299.1125 | 30.0 | 0.0 | 2266.1125 |  | 0.3303 | 0.0 | 0.0 | 0.0 | False | 2 | 0 | 104.0 |  | 30 | 62 | DAP104 I15/N0; DAP111 I15/N0 | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/models/SYA/SYA_lowIC_strict_maskable_dqn_seed0_ckpt100000.pt | benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/daily_outputs/SYA/2023_ckpt100000_strict_maskable_dqn_eval_daily.csv | SYA | SY | 2023 | 0 | 100000 | 100000 |
