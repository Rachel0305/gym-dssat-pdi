# 043_05 SYA lowIC binary-forecast teacher-warmstart MaskablePPO 记录

## 一句话结论

- 分支：`A_binary_forecast_teacher_warmstart_training_completed`
- 本轮使用 041_02 lowIC teacher 作为“时机示范”，并映射到 binary action。
- 本轮使用 043_02 的 30维天气/预报/归一化 observation。

## BC 日志

| bc_epoch | bc_loss | bc_action_accuracy |
| --- | --- | --- |
| 1 | 0.7941 | 0.7031 |
| 2 | 0.7775 | 0.9533 |
| 3 | 0.7734 | 0.9556 |
| 4 | 0.7656 | 0.9495 |
| 5 | 0.7709 | 0.9526 |
| 6 | 0.7467 | 0.9556 |
| 7 | 0.7657 | 0.9526 |
| 8 | 0.7504 | 0.9495 |
| 9 | 0.7238 | 0.9533 |
| 10 | 0.7211 | 0.9526 |
| 11 | 0.7132 | 0.9533 |
| 12 | 0.7045 | 0.9541 |
| 13 | 0.6914 | 0.9526 |
| 14 | 0.688 | 0.9526 |
| 15 | 0.6815 | 0.9495 |
| 16 | 0.6683 | 0.9541 |
| 17 | 0.6612 | 0.9526 |
| 18 | 0.651 | 0.9503 |
| 19 | 0.6352 | 0.9533 |
| 20 | 0.6368 | 0.9503 |

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
| bc_init | 0 | 10 | 1515.3 | 0.5963 |  | 0.0 | 0.0 | 0 | 0 | 1 | 1.0 | 0.5701 |
| ppo_finetune | 1000 | 10 | 1515.3 | 0.5963 |  | 0.0 | 0.0 | 0 | 0 | 1 | 1.0 | 0.5701 |
| ppo_finetune | 2000 | 10 | 1515.3 | 0.5963 |  | 0.0 | 0.0 | 0 | 0 | 1 | 1.0 | 0.5701 |

## 训练模型清单

| station_code | site | seed | checkpoint_step | stage | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 0 | 0 | bc_init | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/models/SYA/SYA_binary_forecast_teacher_warmstart_seed0_bc_init.zip | 5e19dff45f1ab26fabb5ff12df391ec15413edc37ac29ad765a94db764cd136a |
| SYA | SY | 0 | 1000 | ppo_finetune | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/models/SYA/SYA_binary_forecast_teacher_warmstart_seed0_ckpt1000.zip | 7b84c19ab6049d9acd1ba7036fe37546a05cc330493da76f6e273d4601bc4782 |
| SYA | SY | 0 | 2000 | ppo_finetune | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/models/SYA/SYA_binary_forecast_teacher_warmstart_seed0_ckpt2000.zip | 48141ce15c799443bfd9e639191ff8c5e7f42d46327521c0b8e7c45302ff1a7d |

## 逐年验证结果

| station_code | site | year | checkpoint_step | stage | grain_yield_kg_ha | biomass_kg_ha | summary_irrigation_total | summary_nitrogen_total | etcp_mm | WP_ET_kg_m3 | PFP_N_kg_kg | max_swfac | max_nstres | irrigation_event_count | n_event_count | action_sequence | summary_match_score | summary_row_index | daily_csv_path | gap_yield_vs_four_max | gap_wp_et_vs_four_max | gap_pfp_n_vs_four_max | yield_win_vs_four_max | wp_et_win_vs_four_max | pfp_n_win_vs_four_max | any_metric_win_vs_four_max | all3_win_vs_four_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 2014 | 0 | bc_init | 1573.0 | 3214.0 | 0.0 | 0.0 | 295.1 | 0.53 |  | 1.0 | 0.5701 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2014_ckpt0_daily.csv | -9267.3467 | -1.62 |  | False | False | False | False | False |
| SYA | SY | 2015 | 0 | bc_init | 1433.0 | 3407.0 | 0.0 | 0.0 | 298.7 | 0.48 |  | 1.0 | 0.4691 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2015_ckpt0_daily.csv | -9424.8674 | -1.79 |  | False | False | False | False | False |
| SYA | SY | 2016 | 0 | bc_init | 1403.0 | 4128.0 | 0.0 | 0.0 | 348.9 | 0.4 |  | 0.9221 | 0.4696 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2016_ckpt0_daily.csv | -6468.7786 | -1.23 |  | False | False | False | False | False |
| SYA | SY | 2017 | 0 | bc_init | 0.0 | 17.0 | 0.0 | 0.0 | 24.4 |  |  | 0.9604 | 0.0167 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2017_ckpt0_daily.csv | -10896.0376 |  |  | False | False | False | False | False |
| SYA | SY | 2018 | 0 | bc_init | 1540.0 | 3996.0 | 0.0 | 0.0 | 298.8 | 0.52 |  | 0.9752 | 0.4472 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2018_ckpt0_daily.csv | -6554.176 | -1.28 |  | False | False | False | False | False |
| SYA | SY | 2019 | 0 | bc_init | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 |  |  | 0.0 | 0.0 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2019_ckpt0_daily.csv | -10364.5337 |  |  | False | False | False | False | False |
| SYA | SY | 2020 | 0 | bc_init | 1287.0 | 2539.0 | 0.0 | 0.0 | 261.0 | 0.49 |  | 1.0 | 0.4338 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2020_ckpt0_daily.csv | -8583.6927 | -1.61 |  | False | False | False | False | False |
| SYA | SY | 2021 | 0 | bc_init | 1850.0 | 3964.0 | 0.0 | 0.0 | 358.7 | 0.52 |  | 0.9514 | 0.4739 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2021_ckpt0_daily.csv | -8229.975 | -1.62 |  | False | False | False | False | False |
| SYA | SY | 2022 | 0 | bc_init | 3774.0 | 6431.0 | 0.0 | 0.0 | 372.4 | 1.01 |  | 1.0 | 0.4396 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2022_ckpt0_daily.csv | -7148.4487 | -1.52 |  | False | False | False | False | False |
| SYA | SY | 2023 | 0 | bc_init | 2293.0 | 3911.0 | 0.0 | 0.0 | 279.6 | 0.82 |  | 0.9589 | 0.339 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2023_ckpt0_daily.csv | -8643.7651 | -1.45 |  | False | False | False | False | False |
| SYA | SY | 2014 | 1000 | ppo_finetune | 1573.0 | 3214.0 | 0.0 | 0.0 | 295.1 | 0.53 |  | 1.0 | 0.5701 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2014_ckpt1000_daily.csv | -9267.3467 | -1.62 |  | False | False | False | False | False |
| SYA | SY | 2015 | 1000 | ppo_finetune | 1433.0 | 3407.0 | 0.0 | 0.0 | 298.7 | 0.48 |  | 1.0 | 0.4691 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2015_ckpt1000_daily.csv | -9424.8674 | -1.79 |  | False | False | False | False | False |
| SYA | SY | 2016 | 1000 | ppo_finetune | 1403.0 | 4128.0 | 0.0 | 0.0 | 348.9 | 0.4 |  | 0.9221 | 0.4696 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2016_ckpt1000_daily.csv | -6468.7786 | -1.23 |  | False | False | False | False | False |
| SYA | SY | 2017 | 1000 | ppo_finetune | 0.0 | 17.0 | 0.0 | 0.0 | 24.4 |  |  | 0.9604 | 0.0167 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2017_ckpt1000_daily.csv | -10896.0376 |  |  | False | False | False | False | False |
| SYA | SY | 2018 | 1000 | ppo_finetune | 1540.0 | 3996.0 | 0.0 | 0.0 | 298.8 | 0.52 |  | 0.9752 | 0.4472 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2018_ckpt1000_daily.csv | -6554.176 | -1.28 |  | False | False | False | False | False |
| SYA | SY | 2019 | 1000 | ppo_finetune | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 |  |  | 0.0 | 0.0 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2019_ckpt1000_daily.csv | -10364.5337 |  |  | False | False | False | False | False |
| SYA | SY | 2020 | 1000 | ppo_finetune | 1287.0 | 2539.0 | 0.0 | 0.0 | 261.0 | 0.49 |  | 1.0 | 0.4338 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2020_ckpt1000_daily.csv | -8583.6927 | -1.61 |  | False | False | False | False | False |
| SYA | SY | 2021 | 1000 | ppo_finetune | 1850.0 | 3964.0 | 0.0 | 0.0 | 358.7 | 0.52 |  | 0.9514 | 0.4739 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2021_ckpt1000_daily.csv | -8229.975 | -1.62 |  | False | False | False | False | False |
| SYA | SY | 2022 | 1000 | ppo_finetune | 3774.0 | 6431.0 | 0.0 | 0.0 | 372.4 | 1.01 |  | 1.0 | 0.4396 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2022_ckpt1000_daily.csv | -7148.4487 | -1.52 |  | False | False | False | False | False |
| SYA | SY | 2023 | 1000 | ppo_finetune | 2293.0 | 3911.0 | 0.0 | 0.0 | 279.6 | 0.82 |  | 0.9589 | 0.339 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2023_ckpt1000_daily.csv | -8643.7651 | -1.45 |  | False | False | False | False | False |
| SYA | SY | 2014 | 2000 | ppo_finetune | 1573.0 | 3214.0 | 0.0 | 0.0 | 295.1 | 0.53 |  | 1.0 | 0.5701 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2014_ckpt2000_daily.csv | -9267.3467 | -1.62 |  | False | False | False | False | False |
| SYA | SY | 2015 | 2000 | ppo_finetune | 1433.0 | 3407.0 | 0.0 | 0.0 | 298.7 | 0.48 |  | 1.0 | 0.4691 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2015_ckpt2000_daily.csv | -9424.8674 | -1.79 |  | False | False | False | False | False |
| SYA | SY | 2016 | 2000 | ppo_finetune | 1403.0 | 4128.0 | 0.0 | 0.0 | 348.9 | 0.4 |  | 0.9221 | 0.4696 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2016_ckpt2000_daily.csv | -6468.7786 | -1.23 |  | False | False | False | False | False |
| SYA | SY | 2017 | 2000 | ppo_finetune | 0.0 | 17.0 | 0.0 | 0.0 | 24.4 |  |  | 0.9604 | 0.0167 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2017_ckpt2000_daily.csv | -10896.0376 |  |  | False | False | False | False | False |
| SYA | SY | 2018 | 2000 | ppo_finetune | 1540.0 | 3996.0 | 0.0 | 0.0 | 298.8 | 0.52 |  | 0.9752 | 0.4472 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2018_ckpt2000_daily.csv | -6554.176 | -1.28 |  | False | False | False | False | False |
| SYA | SY | 2019 | 2000 | ppo_finetune | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 |  |  | 0.0 | 0.0 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2019_ckpt2000_daily.csv | -10364.5337 |  |  | False | False | False | False | False |
| SYA | SY | 2020 | 2000 | ppo_finetune | 1287.0 | 2539.0 | 0.0 | 0.0 | 261.0 | 0.49 |  | 1.0 | 0.4338 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2020_ckpt2000_daily.csv | -8583.6927 | -1.61 |  | False | False | False | False | False |
| SYA | SY | 2021 | 2000 | ppo_finetune | 1850.0 | 3964.0 | 0.0 | 0.0 | 358.7 | 0.52 |  | 0.9514 | 0.4739 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2021_ckpt2000_daily.csv | -8229.975 | -1.62 |  | False | False | False | False | False |
| SYA | SY | 2022 | 2000 | ppo_finetune | 3774.0 | 6431.0 | 0.0 | 0.0 | 372.4 | 1.01 |  | 1.0 | 0.4396 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2022_ckpt2000_daily.csv | -7148.4487 | -1.52 |  | False | False | False | False | False |
| SYA | SY | 2023 | 2000 | ppo_finetune | 2293.0 | 3911.0 | 0.0 | 0.0 | 279.6 | 0.82 |  | 0.9589 | 0.339 | 0 | 0 |  | 0.0 | 2 | benchmark_results/043_05_sya_lowIC_binary_forecast_teacher_warmstart_maskableppo_smoke2k/daily_outputs/SYA/043_05_SYA_2023_ckpt2000_daily.csv | -8643.7651 | -1.45 |  | False | False | False | False | False |
