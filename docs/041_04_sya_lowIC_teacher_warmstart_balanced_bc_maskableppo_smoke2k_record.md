# 041_04 SYA lowIC teacher warm-start MaskablePPO 记录

## 结论

- 分支：`A_teacher_warmstart_training_completed`
- BC epoch：20
- PPO fine-tune steps：2000
- checkpoint：1000, 2000
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
| ppo_finetune | 1000 | 9779.9 | 2.112 | 48.9 | 225.0 | 200.0 | 10 | 4 | 0.8203 | 0.4732 |
| ppo_finetune | 2000 | 9928.7 | 2.138 | 41.36 | 226.5 | 240.0 | 10 | 5 | 0.8078 | 0.2153 |

## 训练模型清单

| station_code | site | seed | checkpoint_step | stage | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 0 | 0 | bc_init | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_bc_init.zip | f698bca7e6d8dbe43ff86cc357098b242fdedd6bafa78d68411189a440eea6db |
| SYA | SY | 0 | 1000 | ppo_finetune | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt1000.zip | 6d0ffe0dc999d53ba123daf4193672e112884a5dc0476e494d20132a4472804f |
| SYA | SY | 0 | 2000 | ppo_finetune | benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt2000.zip | 99810d842683c130601a5fe069eaf579fa576a72901c5307b04ea5681e501d25 |

## 输出文件

- 训练清单：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/evaluation/041_03_training_checkpoint_inventory.csv`
- 评估总表：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/evaluation/041_03_checkpoint_validation_summary.csv`
- checkpoint 汇总：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/evaluation/041_03_validation_summary_by_checkpoint.csv`
- BC 数据：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/bc_dataset/041_03_bc_dataset.npz`
- JSON：`benchmark_results/041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_smoke2k/041_03_result.json`
