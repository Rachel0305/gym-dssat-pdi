# 046_00 SYA lowIC DSSAT auto 施肥阈值复核记录

- 分支：`B_native_auto_n_not_triggered_by_threshold_variants`
- 性质：零训练 DSSAT baseline 参数复核。
- 年份：[2014, 2017, 2022]
- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 耗时：107.7 s

## 结果摘要

| auto_variant | year | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | nmthr | namnt | ncode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_nmthr50_amt25_fe001 | 2014 | 2820 | 230 | 0 | 0.63 | 50 | 25 | FE001 |
| base_nmthr50_amt25_fe001 | 2017 | 2640 | 273 | 0 | 0.66 | 50 | 25 | FE001 |
| base_nmthr50_amt25_fe001 | 2022 | 4181 | 62 | 0 | 1 | 50 | 25 | FE001 |
| low_nmthr10_amt25_fe001 | 2014 | 2820 | 230 | 0 | 0.63 | 10 | 25 | FE001 |
| low_nmthr10_amt25_fe001 | 2017 | 2640 | 273 | 0 | 0.66 | 10 | 25 | FE001 |
| low_nmthr10_amt25_fe001 | 2022 | 4181 | 62 | 0 | 1 | 10 | 25 | FE001 |
| very_low_nmthr01_amt25_fe001 | 2014 | 2820 | 230 | 0 | 0.63 | 1 | 25 | FE001 |
| very_low_nmthr01_amt25_fe001 | 2017 | 2640 | 273 | 0 | 0.66 | 1 | 25 | FE001 |
| very_low_nmthr01_amt25_fe001 | 2022 | 4181 | 62 | 0 | 1 | 1 | 25 | FE001 |
| high_nmthr99_amt50_fe001 | 2014 | 2820 | 230 | 0 | 0.63 | 99 | 50 | FE001 |
| high_nmthr99_amt50_fe001 | 2017 | 2640 | 273 | 0 | 0.66 | 99 | 50 | FE001 |
| high_nmthr99_amt50_fe001 | 2022 | 4181 | 62 | 0 | 1 | 99 | 50 | FE001 |
| high_nmthr99_amt50_fe005 | 2014 | 2820 | 230 | 0 | 0.63 | 99 | 50 | FE005 |
| high_nmthr99_amt50_fe005 | 2017 | 2640 | 273 | 0 | 0.66 | 99 | 50 | FE005 |
| high_nmthr99_amt50_fe005 | 2022 | 4181 | 62 | 0 | 1 | 99 | 50 | FE005 |

## 失败记录

_空表_

## 输出

- summary: `benchmark_results/046_00_sya_lowIC_dssat_auto_fertilizer_threshold_audit/evaluation/046_00_auto_n_threshold_summary.csv`
- daily: `benchmark_results/046_00_sya_lowIC_dssat_auto_fertilizer_threshold_audit/evaluation/046_00_auto_n_threshold_daily.csv`
- failures: `benchmark_results/046_00_sya_lowIC_dssat_auto_fertilizer_threshold_audit/evaluation/046_00_auto_n_threshold_failures.csv`
