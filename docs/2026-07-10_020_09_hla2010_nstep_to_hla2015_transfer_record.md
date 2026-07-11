# 020_09 HLA2010 n-step DQN → HLA2015 跨年份迁移记录

本轮不重新训练，只加载 HLA2010 n-step 三个成功 checkpoint，在HLA2015环境中直接评估。

## 结果

| scenario | label | train_year | train_seed | train_checkpoint_step | final_gwad | final_cwad | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | Null |  |  |  | 6486.000 | 17168.000 | 0.000 | 0.000 | 1.000 | 0.093 |  |
| dssat_auto | DSSAT auto |  |  |  | 7648.000 | 19021.000 | 141.500 | 0.000 | 0.000 | 0.047 |  |
| expert_2007_shifted | Recorded expert |  |  |  | 7296.000 | 18639.000 | 30.000 | 165.000 | 0.779 | 0.015 |  |
| transfer_2010_seed0 | n-step HLA2010 seed0 | 2010.000 | 0.000 | 30000.000 | 7635.000 | 19059.000 | 45.000 | 0.000 | 0.168 | 0.135 | 1104.410 |
| transfer_2010_seed1 | n-step HLA2010 seed1 | 2010.000 | 1.000 | 10000.000 | 7652.000 | 19061.000 | 120.000 | 0.000 | 0.000 | 0.058 | 1046.082 |
| transfer_2010_seed2 | n-step HLA2010 seed2 | 2010.000 | 2.000 | 20000.000 | 7653.000 | 19070.000 | 120.000 | 0.000 | 0.000 | 0.057 | 1046.725 |

## 输出

- Daily: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_to_hla2015_transfer_020_09/020_09_hla2010_nstep_to_hla2015_daily.csv`
- Summary: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_to_hla2015_transfer_020_09/020_09_hla2010_nstep_to_hla2015_summary.csv`
- Figure: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_to_hla2015_transfer_020_09/figures/020_09_hla2010_nstep_to_hla2015_process.png`

迁移结果只用于判断同站点跨年份泛化，不等同于跨站点泛化。
