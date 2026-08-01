# 042_02 SYA lowIC teacher warm-start MaskablePPO 记录

## 结论

- 分支：`A_teacher_warmstart_training_completed`
- BC epoch：20
- PPO fine-tune steps：100000
- checkpoint：25000, 50000, 75000, 100000
- 训练性质：teacher-assisted same-year feasibility，不是严格跨年泛化。

## BC 训练日志

| bc_epoch | bc_loss | bc_action_accuracy | bc_zero_action_accuracy | bc_nonzero_action_accuracy | bc_pred_nonzero_rate | balanced_nonzero_fraction | nonzero_teacher_samples | zero_teacher_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.4701 | 0.4167 | 0.4727 | 0.3607 | 0.6621 | 0.5 | 73 | 1325 |
| 2 | 1.4667 | 0.4258 | 0.3516 | 0.5 | 0.8242 | 0.5 | 73 | 1325 |
| 3 | 1.4738 | 0.4128 | 0.3268 | 0.4987 | 0.8366 | 0.5 | 73 | 1325 |
| 4 | 1.4538 | 0.4395 | 0.3997 | 0.4792 | 0.8001 | 0.5 | 73 | 1325 |
| 5 | 1.4757 | 0.4141 | 0.3359 | 0.4922 | 0.832 | 0.5 | 73 | 1325 |
| 6 | 1.4237 | 0.4284 | 0.3646 | 0.4922 | 0.8177 | 0.5 | 73 | 1325 |
| 7 | 1.4272 | 0.4421 | 0.3568 | 0.5273 | 0.8216 | 0.5 | 73 | 1325 |
| 8 | 1.388 | 0.4355 | 0.3737 | 0.4974 | 0.8132 | 0.5 | 73 | 1325 |
| 9 | 1.3982 | 0.4023 | 0.3411 | 0.4635 | 0.8294 | 0.5 | 73 | 1325 |
| 10 | 1.363 | 0.4375 | 0.362 | 0.513 | 0.819 | 0.5 | 73 | 1325 |
| 11 | 1.3845 | 0.4284 | 0.375 | 0.4818 | 0.8125 | 0.5 | 73 | 1325 |
| 12 | 1.3559 | 0.418 | 0.3646 | 0.4714 | 0.8177 | 0.5 | 73 | 1325 |
| 13 | 1.3355 | 0.4128 | 0.3451 | 0.4805 | 0.8275 | 0.5 | 73 | 1325 |
| 14 | 1.3301 | 0.4062 | 0.3503 | 0.4622 | 0.8249 | 0.5 | 73 | 1325 |
| 15 | 1.3497 | 0.4115 | 0.3516 | 0.4714 | 0.8242 | 0.5 | 73 | 1325 |
| 16 | 1.3139 | 0.4095 | 0.362 | 0.457 | 0.819 | 0.5 | 73 | 1325 |
| 17 | 1.3054 | 0.4355 | 0.3841 | 0.487 | 0.8079 | 0.5 | 73 | 1325 |
| 18 | 1.3057 | 0.4388 | 0.3737 | 0.5039 | 0.8132 | 0.5 | 73 | 1325 |
| 19 | 1.2866 | 0.4388 | 0.3451 | 0.5326 | 0.8275 | 0.5 | 73 | 1325 |
| 20 | 1.2863 | 0.459 | 0.3542 | 0.5638 | 0.8229 | 0.5 | 73 | 1325 |

## checkpoint 汇总

| stage | checkpoint_step | mean_yield | mean_wp_et | mean_pfp_n | mean_irrigation | mean_nitrogen | any_metric_win_years | all3_win_years | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_init | 0 | 9955.0 | 2.104 | 41.49 | 240.0 | 240.0 | 10 | 2 | 0.8064 | 0.2593 |
| ppo_finetune | 25000 | 3847.4 | 0.893 |  | 225.0 | 0.0 | 0 | 0 | 0.0 | 0.556 |
| ppo_finetune | 50000 | 3847.4 | 0.893 |  | 225.0 | 0.0 | 0 | 0 | 0.0 | 0.556 |
| ppo_finetune | 75000 | 9931.7 | 2.145 | 41.36 | 225.0 | 240.0 | 10 | 6 | 0.8592 | 0.238 |
| ppo_finetune | 100000 | 3847.4 | 0.893 |  | 225.0 | 0.0 | 0 | 0 | 0.0 | 0.556 |

## 训练模型清单

| station_code | site | seed | checkpoint_step | stage | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 0 | 0 | bc_init | benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_bc_init.zip | a50a3d77079158a4c7a6ae4cd734c7b4014474aeb0cf89b5c123fae619b7500f |
| SYA | SY | 0 | 25000 | ppo_finetune | benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt25000.zip | 8fc6ff1454ad86882020d6e60acfc748edfefe431acd371b8db69e2f075ba47f |
| SYA | SY | 0 | 50000 | ppo_finetune | benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt50000.zip | afabdfa3fda66735d7bb812ccb2040ab4ea65d89c5500e34c939f5a57dffb6d3 |
| SYA | SY | 0 | 75000 | ppo_finetune | benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt75000.zip | af2902ad38fe4400d4652f83b6af52d26855cbae3c0aa9de332f0466205858a6 |
| SYA | SY | 0 | 100000 | ppo_finetune | benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt100000.zip | 705bae8e312256811832ed1176aed85e5a02d147ca4a3f2a6440d2ff02363c81 |

## 输出文件

- 训练清单：`benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/evaluation/041_03_training_checkpoint_inventory.csv`
- 评估总表：`benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/evaluation/041_03_checkpoint_validation_summary.csv`
- checkpoint 汇总：`benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/evaluation/041_03_validation_summary_by_checkpoint.csv`
- BC 数据：`benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/bc_dataset/041_03_bc_dataset.npz`
- JSON：`benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/041_03_result.json`
