# 020_10 HLA2010 n-step DQN → HLA2007/2016/2022 跨年份迁移记录

本轮复用015_18/015_19已有null/auto（及2007 recorded）基线，只加载HLA2010 n-step三个成功模型，不重新训练。

## 汇总

| requested_year | scenario | label | train_seed | train_checkpoint_step | final_gwad | final_cwad | rain_total | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007 | null | Null |  |  | 7830.000 | 19528.000 | 339.400 | 0.000 | 0.000 | 0.895 | 0.129 |  |
| 2007 | recorded | Recorded expert |  |  | 7986.000 | 19809.000 | 339.400 | 30.000 | 165.000 | 0.000 | 0.014 |  |
| 2007 | dssat_auto | DSSAT auto |  |  | 7973.000 | 19638.000 | 339.400 | 93.500 | 0.000 | 0.000 | 0.032 |  |
| 2016 | dssat_auto | DSSAT auto |  |  | 7538.000 | 17018.000 | 365.600 | 140.800 | 0.000 | 0.000 | 0.022 |  |
| 2022 | dssat_auto | DSSAT auto |  |  | 7935.000 | 18552.000 | 360.400 | 143.600 | 0.000 | 0.000 | 0.012 |  |
| 2016 | null | Null |  |  | 7224.000 | 16612.000 | 365.600 | 0.000 | 0.000 | 0.690 | 0.150 |  |
| 2022 | null | Null |  |  | 7787.000 | 18370.000 | 360.400 | 0.000 | 0.000 | 0.825 | 0.117 |  |
| 2007 | transfer_2010_seed0 | n-step seed0 | 0 | 30000 | 7987.000 | 19806.000 | 337.600 | 60.000 | 0.000 | 0.000 | 0.074 | 96.733 |
| 2007 | transfer_2010_seed1 | n-step seed1 | 1 | 10000 | 7980.000 | 19690.000 | 337.600 | 120.000 | 0.000 | 0.000 | 0.044 | 29.669 |
| 2007 | transfer_2010_seed2 | n-step seed2 | 2 | 20000 | 7984.000 | 19770.000 | 337.600 | 120.000 | 0.000 | 0.000 | 0.033 | 33.789 |
| 2016 | transfer_2010_seed0 | n-step seed0 | 0 | 30000 | 7538.000 | 17018.000 | 365.600 | 45.000 | 0.000 | 0.000 | 0.022 | 269.467 |
| 2016 | transfer_2010_seed1 | n-step seed1 | 1 | 10000 | 7528.000 | 16955.000 | 365.600 | 120.000 | 50.000 | 0.000 | 0.022 | -66.393 |
| 2016 | transfer_2010_seed2 | n-step seed2 | 2 | 20000 | 7538.000 | 17018.000 | 365.600 | 120.000 | 0.000 | 0.000 | 0.022 | 194.467 |
| 2022 | transfer_2010_seed0 | n-step seed0 | 0 | 30000 | 7935.000 | 18552.000 | 356.400 | 60.000 | 0.000 | 0.000 | 0.023 | 87.681 |
| 2022 | transfer_2010_seed1 | n-step seed1 | 1 | 10000 | 7932.000 | 18525.000 | 356.400 | 120.000 | 0.000 | 0.000 | 0.012 | 24.951 |
| 2022 | transfer_2010_seed2 | n-step seed2 | 2 | 20000 | 7925.000 | 18455.000 | 356.400 | 120.000 | 0.000 | 0.000 | 0.012 | 18.308 |

## 输出

- Daily: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_transfer_2007_2016_2022_020_10/020_10_hla2010_nstep_transfer_daily.csv`
- Summary: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_transfer_2007_2016_2022_020_10/020_10_hla2010_nstep_transfer_summary.csv`
- Figures: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_transfer_2007_2016_2022_020_10/figures`

结果仅用于HLA站点内跨年份迁移，不等同于跨站点泛化。

## 结果判读

- HLA2007：三个迁移 seed 的产量均高于 DSSAT auto（7973 kg/ha），分别为7987、7980和7984 kg/ha；灌溉为60、120和120 mm，均不施氮。
- HLA2016：seed0与seed2追平 DSSAT auto（7538 kg/ha），seed1为7528 kg/ha；seed0只用45 mm水，seed2用120 mm且不施氮，seed1额外用了50 kg/ha氮。
- HLA2022：seed0追平 auto（7935 kg/ha），seed1和seed2略低（7932和7925 kg/ha）；三个 seed 均不施氮，灌溉为60、120和120 mm。
- 三个年份所有迁移结果均明显高于各自 null，说明HLA2010 n-step策略具有较强的同站点跨年份泛化迹象；但2016和2022仍存在少量 seed 间产量差异，不能称完全一致。
