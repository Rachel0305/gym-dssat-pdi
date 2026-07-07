# 017_08 SY 本地 DQN 训练与跨年迁移记录

## 目的

017_07 证明 HL/YC 模型无法直接迁移到 SY，因为 observation space 维度不一致。本轮改为同一方法在 SY 本地训练一年，再迁移到 SY 其他年份。

## 基准筛选

| year | scenario | final_gwad | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | auto_gain_vs_null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | dssat_auto | 7016.00 | 33.00 | 0.00 | 0.00 | 0.60 | 14.00 |
| 2012 | null | 7002.00 | 0.00 | 0.00 | 0.00 | 0.60 | 14.00 |
| 2012 | recorded | 10166.00 | 0.00 | 293.00 | 0.00 | 0.02 | 14.00 |
| 2014 | dssat_auto | 2724.00 | 33.40 | 0.00 | 0.00 | 0.44 | -45.00 |
| 2014 | null | 2769.00 | 0.00 | 0.00 | 0.00 | 0.44 | -45.00 |
| 2014 | recorded | 9593.00 | 0.00 | 293.00 | 0.81 | 0.01 | -45.00 |

- 基准图：`DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/figures/017_08_sy_baseline_screen.png`

## 训练年选择

- 训练年：SY2014
- 选择理由：在已筛选年份中，管理基准相对 null 的增产空间最大。

## checkpoint 训练结果

| year | scenario | checkpoint | final_gwad | irrigation_total | fertilizer_total | total_reward | yield_diff_vs_null | yield_diff_vs_dssat_auto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | dqn_ckpt5000 | 5000.00 | 10701.00 | 120.00 | 300.00 | 6311.63 | 7932.00 | 7977.00 |
| 2014 | dqn_ckpt10000 | 10000.00 | 11088.00 | 120.00 | 300.00 | 6698.68 | 8319.00 | 8364.00 |
| 2014 | dqn_ckpt15000 | 15000.00 | 11216.00 | 120.00 | 300.00 | 6826.54 | 8447.00 | 8492.00 |
| 2014 | dqn_ckpt20000 | 20000.00 | 2769.00 | 75.00 | 100.00 | -575.00 | 0.00 | 45.00 |
| 2014 | dqn_ckpt25000 | 25000.00 | 2769.00 | 90.00 | 150.00 | -840.00 | 0.00 | 45.00 |
| 2014 | dqn_ckpt30000 | 30000.00 | 2769.00 | 0.00 | 0.00 | 0.00 | 0.00 | 45.00 |
| 2014 | dqn_ckpt35000 | 35000.00 | 2769.00 | 0.00 | 0.00 | 0.00 | 0.00 | 45.00 |
| 2014 | dqn_ckpt40000 | 40000.00 | 2769.00 | 0.00 | 0.00 | 0.00 | 0.00 | 45.00 |
| 2014 | dqn_ckpt45000 | 45000.00 | 2769.00 | 0.00 | 0.00 | 0.00 | 0.00 | 45.00 |
| 2014 | dqn_ckpt50000 | 50000.00 | 2769.00 | 0.00 | 0.00 | 0.00 | 0.00 | 45.00 |

## 跨年迁移结果

| year | scenario | checkpoint | final_gwad | irrigation_total | fertilizer_total | total_reward | yield_diff_vs_null | yield_diff_vs_recorded | yield_diff_vs_dssat_auto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | transfer_SY2014_ckpt15000 | 15000.00 | 10239.00 | 120.00 | 300.00 | 1617.06 | 3237.00 | 73.00 | 3223.00 |
| 2014 | transfer_SY2014_ckpt15000 | 15000.00 | 11216.00 | 120.00 | 300.00 | 6826.54 | 8447.00 | 1623.00 | 8492.00 |

- 迁移总图：`DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/figures/017_08_sy_cross_year_transfer_summary.png`
