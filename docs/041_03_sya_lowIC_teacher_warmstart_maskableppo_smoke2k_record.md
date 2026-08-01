# 041_03 SYA lowIC teacher warm-start MaskablePPO 记录

## 结论

- 分支：`A_teacher_warmstart_training_completed`
- BC epoch：20
- PPO fine-tune steps：2000
- checkpoint：1000, 2000
- 训练性质：teacher-assisted same-year feasibility，不是严格跨年泛化。

## BC 训练日志

| bc_epoch | bc_loss | bc_action_accuracy |
| --- | --- | --- |
| 1 | 1.1261 | 0.5736 |
| 2 | 1.0964 | 0.9462 |
| 3 | 1.0811 | 0.951 |
| 4 | 1.0778 | 0.9441 |
| 5 | 1.0818 | 0.9471 |
| 6 | 1.0492 | 0.9517 |
| 7 | 1.0733 | 0.9479 |
| 8 | 1.0543 | 0.9456 |
| 9 | 1.0196 | 0.9487 |
| 10 | 1.0126 | 0.9487 |
| 11 | 1.0051 | 0.9487 |
| 12 | 0.9918 | 0.9487 |
| 13 | 0.9731 | 0.9479 |
| 14 | 0.9695 | 0.9487 |
| 15 | 0.9549 | 0.9456 |
| 16 | 0.9423 | 0.9502 |
| 17 | 0.9306 | 0.9487 |
| 18 | 0.9075 | 0.9464 |
| 19 | 0.8913 | 0.9487 |
| 20 | 0.8821 | 0.9464 |

## checkpoint 汇总

| stage | checkpoint_step | mean_yield | mean_wp_et | mean_pfp_n | mean_irrigation | mean_nitrogen | any_metric_win_years | all3_win_years | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_init | 0 | 1515.3 | 0.5963 |  | 0.0 | 0.0 | 0 | 0 | 1.0 | 0.5701 |
| ppo_finetune | 1000 | 1515.3 | 0.5963 |  | 0.0 | 0.0 | 0 | 0 | 1.0 | 0.5701 |
| ppo_finetune | 2000 | 1486.9 | 0.5612 |  | 9.0 | 0.0 | 0 | 0 | 1.0 | 0.5701 |

## 训练模型清单

| station_code | site | seed | checkpoint_step | stage | model_path | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 0 | 0 | bc_init | benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_bc_init.zip | b9213c064eb7f2be42e32cd78e20884c7d717d7d31c6d5d44a18bb05cf63efe3 |
| SYA | SY | 0 | 1000 | ppo_finetune | benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt1000.zip | a0ae899558947cf116c274bf65b21d4a7969bbe0b1f952b979959fd8ad0c4f7e |
| SYA | SY | 0 | 2000 | ppo_finetune | benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/models/SYA/SYA_teacher_warmstart_maskableppo_seed0_ckpt2000.zip | dd76bb7abd67cde76e47e59bdac1a9ff7340dbbaa69785519a49a7de2780317b |

## 输出文件

- 训练清单：`benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/evaluation/041_03_training_checkpoint_inventory.csv`
- 评估总表：`benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/evaluation/041_03_checkpoint_validation_summary.csv`
- checkpoint 汇总：`benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/evaluation/041_03_validation_summary_by_checkpoint.csv`
- BC 数据：`benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/bc_dataset/041_03_bc_dataset.npz`
- JSON：`benchmark_results/041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k/041_03_result.json`
