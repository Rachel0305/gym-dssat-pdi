# 041_04 SYA lowIC teacher warm-start MaskablePPO 记录

## 结论

- 分支：`A_teacher_warmstart_training_completed`
- BC epoch：20
- PPO fine-tune steps：100000
- checkpoint：25000, 50000, 75000, 100000
- 训练性质：teacher-assisted same-year feasibility，不是严格跨年泛化。

## BC 训练日志

| bc_epoch | bc_loss | bc_action_accuracy | bc_zero_action_accuracy | bc_nonzero_action_accuracy | bc_pred_nonzero_rate | balanced_nonzero_fraction | nonzero_teacher_samples | zero_teacher_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.4671 | 0.3991 | 0.3411 | 0.457 | 0.8294 | 0.5 | 73 | 1325 |
| 2 | 1.4605 | 0.4395 | 0.3516 | 0.5273 | 0.8242 | 0.5 | 73 | 1325 |
| 3 | 1.4649 | 0.4323 | 0.3268 | 0.5378 | 0.8366 | 0.5 | 73 | 1325 |
| 4 | 1.4415 | 0.4766 | 0.3997 | 0.5534 | 0.8001 | 0.5 | 73 | 1325 |
| 5 | 1.4631 | 0.4303 | 0.3359 | 0.5247 | 0.832 | 0.5 | 73 | 1325 |
| 6 | 1.4085 | 0.4316 | 0.3646 | 0.4987 | 0.8177 | 0.5 | 73 | 1325 |
| 7 | 1.4106 | 0.4434 | 0.3568 | 0.5299 | 0.8216 | 0.5 | 73 | 1325 |
| 8 | 1.3693 | 0.4499 | 0.3737 | 0.526 | 0.8132 | 0.5 | 73 | 1325 |
| 9 | 1.3767 | 0.4258 | 0.3411 | 0.5104 | 0.8294 | 0.5 | 73 | 1325 |
| 10 | 1.3435 | 0.4486 | 0.362 | 0.5352 | 0.819 | 0.5 | 73 | 1325 |
| 11 | 1.3608 | 0.4473 | 0.375 | 0.5195 | 0.8125 | 0.5 | 73 | 1325 |
| 12 | 1.3342 | 0.4408 | 0.3646 | 0.5169 | 0.8177 | 0.5 | 73 | 1325 |
| 13 | 1.3143 | 0.4206 | 0.3451 | 0.4961 | 0.8275 | 0.5 | 73 | 1325 |
| 14 | 1.3079 | 0.444 | 0.3503 | 0.5378 | 0.8249 | 0.5 | 73 | 1325 |
| 15 | 1.3253 | 0.4355 | 0.3516 | 0.5195 | 0.8242 | 0.5 | 73 | 1325 |
| 16 | 1.2892 | 0.431 | 0.362 | 0.5 | 0.819 | 0.5 | 73 | 1325 |
| 17 | 1.2834 | 0.4505 | 0.3841 | 0.5169 | 0.8079 | 0.5 | 73 | 1325 |
| 18 | 1.284 | 0.4453 | 0.3737 | 0.5169 | 0.8132 | 0.5 | 73 | 1325 |
| 19 | 1.2685 | 0.4349 | 0.3451 | 0.5247 | 0.8275 | 0.5 | 73 | 1325 |
| 20 | 1.262 | 0.446 | 0.3542 | 0.5378 | 0.8229 | 0.5 | 73 | 1325 |

## checkpoint 汇总

| stage | checkpoint_step | mean_yield | mean_wp_et | mean_pfp_n | mean_irrigation | mean_nitrogen | any_metric_win_years | all3_win_years | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_init | 0 | 9960.8 | 2.1 | 41.51 | 240.0 | 240.0 | 10 | 2 | 0.8493 | 0.1897 |
| ppo_finetune | 25000 | 9512.4 | 2.021 | 47.56 | 225.0 | 200.0 | 10 | 2 | 0.9662 | 0.4319 |
| ppo_finetune | 50000 | 7209.1 | 1.647 | 36.04 | 135.0 | 200.0 | 6 | 2 | 1.0 | 0.251 |
| ppo_finetune | 75000 | 9595.0 | 2.041 | 47.99 | 225.0 | 200.0 | 10 | 3 | 0.9053 | 0.4313 |
| ppo_finetune | 100000 | 9595.0 | 2.041 | 47.99 | 225.0 | 200.0 | 10 | 3 | 0.9053 | 0.4313 |

## 训练模型清单

| station_code | site | seed | checkpoint_step | stage | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 0 | 0 | bc_init | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_bc_init.zip | 87e470a9a625532da3c3277d7ecc269693c5435a32e394135c86dc41775951c4 |
| SYA | SY | 0 | 25000 | ppo_finetune | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt25000.zip | 8dae098f70ec7b68976f811d6acf6f15ce08fa0046571a2bf2f508eb8d447dd3 |
| SYA | SY | 0 | 50000 | ppo_finetune | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt50000.zip | 821bd52796e6cb32124d3281d909b1dd814bcbab6163571a05044edb1e5603e7 |
| SYA | SY | 0 | 75000 | ppo_finetune | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt75000.zip | 693419c2915724d413d19283138c0033ac3e6c9f906d365be19c10d56166906f |
| SYA | SY | 0 | 100000 | ppo_finetune | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt100000.zip | 7e7ea316c1339e5cc5efca00aaf39497fde319f7ff4c7659712ad65d42cfee99 |

## 输出文件

- 训练清单：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/evaluation/041_03_training_checkpoint_inventory.csv`
- 评估总表：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/evaluation/041_03_checkpoint_validation_summary.csv`
- checkpoint 汇总：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/evaluation/041_03_validation_summary_by_checkpoint.csv`
- BC 数据：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/bc_dataset/041_03_bc_dataset.npz`
- JSON：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo/041_03_result.json`
