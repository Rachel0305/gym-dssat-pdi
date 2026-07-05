# 016_07 FQ 施氮时机响应诊断

固定灌溉为 `I60 critical`，固定总施氮为 `N150`，只改变施氮时机。

| year | timing_mode | status | final_gwad | final_cwad | max_water_stress | max_nitrogen_stress | last_dap | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2008 | early | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2008/I60_N150_early |
| 2008 | critical | ok | 7890.000 | 13535.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2008/I60_N150_critical |
| 2008 | late | ok | 7890.000 | 13535.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2008/I60_N150_late |
| 2008 | split_early_late | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2008/I60_N150_split_early_late |
| 2008 | single_critical | ok | 7890.000 | 13535.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2008/I60_N150_single_critical |
| 2016 | early | ok | 8012.000 | 14082.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2016/I60_N150_early |
| 2016 | critical | ok | 8012.000 | 14083.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2016/I60_N150_critical |
| 2016 | late | ok | 8012.000 | 14085.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2016/I60_N150_late |
| 2016 | split_early_late | ok | 8012.000 | 14081.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2016/I60_N150_split_early_late |
| 2016 | single_critical | ok | 8012.000 | 14085.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_timing_response_diagnostic_016_07/runs/2016/I60_N150_single_critical |

## 初步判读

- FQ2008: 最佳产量 `7890.0`，对应时机模式 `early, critical, late, split_early_late, single_critical`。
- FQ2016: 最佳产量 `8012.0`，对应时机模式 `early, critical, late, split_early_late, single_critical`。
