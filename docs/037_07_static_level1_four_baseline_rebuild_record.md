# 037_07 静态 level-1 四基线重建记录

- 模式：`smoke`
- 耗时：134.5 秒
- 成功情景：20 / 20
- 非 ok 记录：0

## 本轮修复

旧 034_00 把 `@I IDATE` / `@F FDATE` 行首编号误写成事件序号。DSSAT 将该列解释为 management level，因此只会把 treatment 选中的 level 1 事件写入 `DSSAT48.INP`。本轮将同一情景内所有静态应用事件统一写为 level 1，以复现多次 recorded/expert 管理事件。

## 覆盖状态

| scenario | status | n |
| --- | --- | --- |
| dssat_auto | ok | 5 |
| null | ok | 5 |
| official_extension_expert | ok | 5 |
| recorded_farmer_template | ok | 5 |

## 非 ok 记录

无记录。

## 站点均值预览

| station_code | scenario | n | mean_yield | mean_irrigation | mean_nitrogen | mean_wp_et | mean_pfp_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | dssat_auto | 1 | 7972.7026 | 0.0 | 0.0 | 2.58 |  |
| FQA | null | 1 | 7972.7026 | 0.0 | 0.0 | 2.58 |  |
| FQA | official_extension_expert | 1 | 7895.3009 | 229.0 | 245.0 | 2.35 | 32.2 |
| FQA | recorded_farmer_template | 1 | 7961.8384 | 75.0 | 144.0 | 2.53 | 55.3 |
| HLA | dssat_auto | 1 | 9636.0626 | 333.0 | 0.0 | 2.15 |  |
| HLA | null | 1 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| HLA | official_extension_expert | 1 | 9620.0287 | 266.0 | 297.0 | 2.11 | 32.4 |
| HLA | recorded_farmer_template | 1 | 3979.9664 | 60.0 | 330.0 | 1.44 | 12.1 |
| LCA | dssat_auto | 1 | 8812.0013 | 89.0 | 0.0 | 2.84 |  |
| LCA | null | 1 | 8821.4655 | 0.0 | 0.0 | 3.03 |  |
| LCA | official_extension_expert | 1 | 8807.785 | 199.0 | 245.0 | 2.86 | 36.6 |
| LCA | recorded_farmer_template | 1 | 8805.9845 | 130.0 | 250.0 | 2.85 | 35.8 |
| SYA | dssat_auto | 1 | 6868.0969 | 0.0 | 0.0 | 1.5 |  |
| SYA | null | 1 | 6868.0969 | 0.0 | 0.0 | 1.5 |  |
| SYA | official_extension_expert | 1 | 9048.8629 | 266.0 | 297.0 | 1.93 | 30.5 |
| SYA | recorded_farmer_template | 1 | 9681.2201 | 0.0 | 586.0 | 2.1 | 16.5 |
| YCA | dssat_auto | 1 | 8521.8512 | 215.0 | 0.0 | 2.59 |  |
| YCA | null | 1 | 4029.343 | 0.0 | 0.0 | 1.89 |  |
| YCA | official_extension_expert | 1 | 8591.792 | 229.0 | 245.0 | 2.59 | 35.1 |
| YCA | recorded_farmer_template | 1 | 8591.792 | 120.0 | 374.0 | 2.76 | 23.0 |

## 输出文件

- summary: `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_smoke_baseline_summary.csv`
- daily: `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_smoke_baseline_daily.csv`
- audit: `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_smoke_management_event_audit.csv`
- manifest: `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_smoke_coverage_manifest.csv`
- failures: `benchmark_results/037_07_static_level1_four_baseline_rebuild/evaluation/037_07_smoke_failures.csv`
