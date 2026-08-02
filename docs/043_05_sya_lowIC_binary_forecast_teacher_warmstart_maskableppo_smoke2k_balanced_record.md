# 043_05 SYA lowIC binary-forecast teacher-warmstart MaskablePPO 记录

## 一句话结论

- 分支：`A_binary_forecast_teacher_warmstart_training_completed`
- 本轮使用 041_02 lowIC teacher 作为“时机示范”，并映射到 binary action。
- 本轮使用 043_02 的 30维天气/预报/归一化 observation。

## BC 日志

| bc_epoch | bc_loss | bc_action_accuracy | bc_nonzero_batch_action_accuracy | bc_full_nonzero_action_accuracy | bc_full_nonzero_pred_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.153 | 0.4025 | 0.5534 | 0.5672 | 0.628 |
| 2 | 1.0927 | 0.4021 | 0.559 | 0.5672 | 0.628 |
| 3 | 1.1177 | 0.4029 | 0.4684 | 0.5672 | 0.628 |
| 4 | 1.1055 | 0.4067 | 0.5582 | 0.5672 | 0.628 |
| 5 | 1.078 | 0.3967 | 0.5863 | 0.6269 | 0.628 |
| 6 | 1.0604 | 0.404 | 0.7264 | 0.6269 | 0.628 |
| 7 | 1.1066 | 0.3947 | 0.6152 | 0.6866 | 0.628 |
| 8 | 1.1026 | 0.4029 | 0.6676 | 0.6567 | 0.628 |
| 9 | 1.0181 | 0.407 | 0.6989 | 0.6418 | 0.628 |
| 10 | 1.0473 | 0.4016 | 0.644 | 0.6567 | 0.628 |
| 11 | 1.0713 | 0.4037 | 0.6673 | 0.6567 | 0.628 |
| 12 | 1.0747 | 0.3992 | 0.6261 | 0.6567 | 0.628 |
| 13 | 1.0136 | 0.4105 | 0.7065 | 0.7463 | 0.628 |
| 14 | 0.9885 | 0.4055 | 0.7712 | 0.7313 | 0.628 |
| 15 | 1.0196 | 0.4123 | 0.7364 | 0.7313 | 0.628 |
| 16 | 1.0234 | 0.4017 | 0.7255 | 0.7313 | 0.628 |
| 17 | 1.0041 | 0.4032 | 0.7255 | 0.7313 | 0.628 |
| 18 | 1.0033 | 0.4084 | 0.7294 | 0.6866 | 0.628 |
| 19 | 0.9893 | 0.4045 | 0.6296 | 0.6567 | 0.628 |
| 20 | 0.9877 | 0.4091 | 0.6694 | 0.6716 | 0.628 |

## BC 数据摘要

| teacher_tier | is_nonzero_action_day | samples | mean_weight |
| --- | --- | --- | --- |
| near_miss | False | 668 | 0.1 |
| near_miss | True | 32 | 0.5 |
| strong_all3 | False | 663 | 0.2 |
| strong_all3 | True | 35 | 1.0 |

## checkpoint 汇总

| stage | checkpoint_step | validation_years | mean_yield | mean_wp_et | mean_pfp_n | mean_irrigation | mean_nitrogen | any_metric_win_years | all3_win_years | unique_action_signatures | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_init | 0 | 10 | 9634.2 | 2.013 | 40.14 | 225.0 | 240.0 | 9 | 2 | 1 | 1.0 | 0.1409 |
| ppo_finetune | 1000 | 10 | 9634.2 | 2.013 | 40.14 | 225.0 | 240.0 | 9 | 2 | 1 | 1.0 | 0.1409 |
| ppo_finetune | 2000 | 10 | 6427.7 | 1.57 | 26.79 | 90.0 | 240.0 | 3 | 1 | 1 | 1.0 | 0.0433 |

## 训练模型清单

| station_code | site | seed | checkpoint_step | stage | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 0 | 0 | bc_init | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/models/SYA/SYA_binary_forecast_teacher_warmstart_seed0_bc_init.zip | c2773e1efac2135ba33275643001c488b66e881f1dc08deb63c9603a2b20b97a |
| SYA | SY | 0 | 1000 | ppo_finetune | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/models/SYA/SYA_binary_forecast_teacher_warmstart_seed0_ckpt1000.zip | f3ba2464b9e52bc204e24256274ae2681b6259349325f9b4d50ca673e944e8c3 |
| SYA | SY | 0 | 2000 | ppo_finetune | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/models/SYA/SYA_binary_forecast_teacher_warmstart_seed0_ckpt2000.zip | a70b8174adacbbf6a5bedd956899204c70063fc6726e61a04864eb17bfd9c67f |

## 逐年验证结果

| station_code | site | year | checkpoint_step | stage | grain_yield_kg_ha | biomass_kg_ha | summary_irrigation_total | summary_nitrogen_total | etcp_mm | WP_ET_kg_m3 | PFP_N_kg_kg | max_swfac | max_nstres | irrigation_event_count | n_event_count | action_sequence | summary_match_score | summary_row_index | daily_csv_path | gap_yield_vs_four_max | gap_wp_et_vs_four_max | gap_pfp_n_vs_four_max | yield_win_vs_four_max | wp_et_win_vs_four_max | pfp_n_win_vs_four_max | any_metric_win_vs_four_max | all3_win_vs_four_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 2014 | 0 | bc_init | 9693.0 | 17646.0 | 225.0 | 240.0 | 479.9 | 2.02 | 40.4 | 0.9069 | 0.1409 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2014_ckpt0_daily.csv | -1147.3467 | -0.13 | 3.9 | False | False | True | True | False |
| SYA | SY | 2015 | 0 | bc_init | 10978.0 | 18597.0 | 225.0 | 240.0 | 496.8 | 2.21 | 45.7 | 0.0 | 0.0713 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2015_ckpt0_daily.csv | 120.1326 | -0.06 | 9.1 | True | False | True | True | False |
| SYA | SY | 2016 | 0 | bc_init | 7981.0 | 15473.0 | 225.0 | 240.0 | 479.7 | 1.66 | 33.3 | 0.0 | 0.0152 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2016_ckpt0_daily.csv | 109.2214 | 0.03 | 6.8 | True | True | True | True | True |
| SYA | SY | 2017 | 0 | bc_init | 7696.0 | 15415.0 | 225.0 | 240.0 | 411.5 | 1.87 | 32.1 | 1.0 | 0.0144 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2017_ckpt0_daily.csv | -3200.0376 | -0.58 | -4.6 | False | False | False | False | False |
| SYA | SY | 2018 | 0 | bc_init | 8214.0 | 15052.0 | 225.0 | 240.0 | 475.4 | 1.73 | 34.2 | 0.0 | 0.0241 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2018_ckpt0_daily.csv | 119.824 | -0.07 | 6.9 | True | False | True | True | False |
| SYA | SY | 2019 | 0 | bc_init | 10394.0 | 18196.0 | 225.0 | 240.0 | 510.6 | 2.04 | 43.3 | 0.0 | 0.013 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2019_ckpt0_daily.csv | 29.4663 | -0.08 | 8.4 | True | False | True | True | False |
| SYA | SY | 2020 | 0 | bc_init | 9906.0 | 18070.0 | 225.0 | 240.0 | 481.0 | 2.06 | 41.3 | 0.3394 | 0.0162 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2020_ckpt0_daily.csv | 35.3073 | -0.04 | 8.1 | True | False | True | True | False |
| SYA | SY | 2021 | 0 | bc_init | 10250.0 | 16860.0 | 225.0 | 240.0 | 474.1 | 2.16 | 42.7 | 0.0 | 0.0136 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2021_ckpt0_daily.csv | 170.025 | 0.02 | 8.8 | True | True | True | True | True |
| SYA | SY | 2022 | 0 | bc_init | 10544.0 | 16418.0 | 225.0 | 240.0 | 487.1 | 2.16 | 43.9 | 0.0 | 0.0122 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2022_ckpt0_daily.csv | -378.4487 | -0.37 | 7.1 | False | False | True | True | False |
| SYA | SY | 2023 | 0 | bc_init | 10686.0 | 18426.0 | 225.0 | 240.0 | 481.1 | 2.22 | 44.5 | 0.0 | 0.016 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2023_ckpt0_daily.csv | -250.7651 | -0.05 | 7.7 | False | False | True | True | False |
| SYA | SY | 2014 | 1000 | ppo_finetune | 9693.0 | 17646.0 | 225.0 | 240.0 | 479.9 | 2.02 | 40.4 | 0.9069 | 0.1409 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2014_ckpt1000_daily.csv | -1147.3467 | -0.13 | 3.9 | False | False | True | True | False |
| SYA | SY | 2015 | 1000 | ppo_finetune | 10978.0 | 18597.0 | 225.0 | 240.0 | 496.8 | 2.21 | 45.7 | 0.0 | 0.0713 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2015_ckpt1000_daily.csv | 120.1326 | -0.06 | 9.1 | True | False | True | True | False |
| SYA | SY | 2016 | 1000 | ppo_finetune | 7981.0 | 15473.0 | 225.0 | 240.0 | 479.7 | 1.66 | 33.3 | 0.0 | 0.0152 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2016_ckpt1000_daily.csv | 109.2214 | 0.03 | 6.8 | True | True | True | True | True |
| SYA | SY | 2017 | 1000 | ppo_finetune | 7696.0 | 15415.0 | 225.0 | 240.0 | 411.5 | 1.87 | 32.1 | 1.0 | 0.0144 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2017_ckpt1000_daily.csv | -3200.0376 | -0.58 | -4.6 | False | False | False | False | False |
| SYA | SY | 2018 | 1000 | ppo_finetune | 8214.0 | 15052.0 | 225.0 | 240.0 | 475.4 | 1.73 | 34.2 | 0.0 | 0.0241 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2018_ckpt1000_daily.csv | 119.824 | -0.07 | 6.9 | True | False | True | True | False |
| SYA | SY | 2019 | 1000 | ppo_finetune | 10394.0 | 18196.0 | 225.0 | 240.0 | 510.6 | 2.04 | 43.3 | 0.0 | 0.013 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2019_ckpt1000_daily.csv | 29.4663 | -0.08 | 8.4 | True | False | True | True | False |
| SYA | SY | 2020 | 1000 | ppo_finetune | 9906.0 | 18070.0 | 225.0 | 240.0 | 481.0 | 2.06 | 41.3 | 0.3394 | 0.0162 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2020_ckpt1000_daily.csv | 35.3073 | -0.04 | 8.1 | True | False | True | True | False |
| SYA | SY | 2021 | 1000 | ppo_finetune | 10250.0 | 16860.0 | 225.0 | 240.0 | 474.1 | 2.16 | 42.7 | 0.0 | 0.0136 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2021_ckpt1000_daily.csv | 170.025 | 0.02 | 8.8 | True | True | True | True | True |
| SYA | SY | 2022 | 1000 | ppo_finetune | 10544.0 | 16418.0 | 225.0 | 240.0 | 487.1 | 2.16 | 43.9 | 0.0 | 0.0122 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2022_ckpt1000_daily.csv | -378.4487 | -0.37 | 7.1 | False | False | True | True | False |
| SYA | SY | 2023 | 1000 | ppo_finetune | 10686.0 | 18426.0 | 225.0 | 240.0 | 481.1 | 2.22 | 44.5 | 0.0 | 0.016 | 5 | 3 | DAP0 I0/N80; DAP0 I45/N0; DAP9 I0/N80; DAP10 I45/N0; DAP16 I0/N80; DAP17 I45/N0; DAP24 I45/N0; DAP31 I45/N0 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2023_ckpt1000_daily.csv | -250.7651 | -0.05 | 7.7 | False | False | True | True | False |
| SYA | SY | 2014 | 2000 | ppo_finetune | 3993.0 | 10374.0 | 90.0 | 240.0 | 354.8 | 1.13 | 16.6 | 1.0 | 0.0122 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2014_ckpt2000_daily.csv | -6847.3467 | -1.02 | -19.9 | False | False | False | False | False |
| SYA | SY | 2015 | 2000 | ppo_finetune | 3325.0 | 8764.0 | 90.0 | 240.0 | 410.0 | 0.81 | 13.9 | 0.9441 | 0.0122 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2015_ckpt2000_daily.csv | -7532.8674 | -1.46 | -22.7 | False | False | False | False | False |
| SYA | SY | 2016 | 2000 | ppo_finetune | 8033.0 | 15683.0 | 90.0 | 240.0 | 465.5 | 1.73 | 33.5 | 0.0 | 0.0152 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2016_ckpt2000_daily.csv | 161.2214 | 0.1 | 7.0 | True | True | True | True | True |
| SYA | SY | 2017 | 2000 | ppo_finetune | 3550.0 | 6930.0 | 90.0 | 240.0 | 274.3 | 1.29 | 14.8 | 1.0 | 0.0144 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2017_ckpt2000_daily.csv | -7346.0376 | -1.16 | -21.9 | False | False | False | False | False |
| SYA | SY | 2018 | 2000 | ppo_finetune | 5956.0 | 13101.0 | 90.0 | 240.0 | 395.2 | 1.51 | 24.8 | 0.9481 | 0.017 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2018_ckpt2000_daily.csv | -2138.176 | -0.29 | -2.5 | False | False | False | False | False |
| SYA | SY | 2019 | 2000 | ppo_finetune | 6044.0 | 12272.0 | 90.0 | 240.0 | 419.1 | 1.44 | 25.2 | 0.9439 | 0.013 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2019_ckpt2000_daily.csv | -4320.5337 | -0.68 | -9.7 | False | False | False | False | False |
| SYA | SY | 2020 | 2000 | ppo_finetune | 4364.0 | 8461.0 | 90.0 | 240.0 | 357.5 | 1.22 | 18.2 | 0.986 | 0.0433 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2020_ckpt2000_daily.csv | -5506.6927 | -0.88 | -15.0 | False | False | False | False | False |
| SYA | SY | 2021 | 2000 | ppo_finetune | 10022.0 | 16998.0 | 90.0 | 240.0 | 447.7 | 2.24 | 41.8 | 0.5357 | 0.0136 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2021_ckpt2000_daily.csv | -57.975 | 0.1 | 7.9 | False | True | True | True | False |
| SYA | SY | 2022 | 2000 | ppo_finetune | 11067.0 | 17664.0 | 90.0 | 240.0 | 458.4 | 2.41 | 46.1 | 0.0 | 0.0122 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2022_ckpt2000_daily.csv | 144.5513 | -0.12 | 9.3 | True | False | True | True | False |
| SYA | SY | 2023 | 2000 | ppo_finetune | 7923.0 | 14823.0 | 90.0 | 240.0 | 412.5 | 1.92 | 33.0 | 0.9092 | 0.016 | 2 | 3 | DAP0 I45/N80; DAP9 I45/N80; DAP16 I0/N80 | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k_balanced/daily_outputs/SYA/043_05_SYA_2023_ckpt2000_daily.csv | -3013.7651 | -0.35 | -3.8 | False | False | False | False | False |
