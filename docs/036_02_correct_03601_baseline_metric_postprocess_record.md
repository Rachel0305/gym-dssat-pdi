# 036_02 修正 036_01 基线与指标后处理记录

## 结论先说

- 036_02 不训练、不改模型，只修正 036_01 的比较后处理。
- SYA 已通过 `031_29_sy_baseline_summary.csv` 接入可用基线；但 SYA recorded farmer 是多个 template，不是单一原始 recorded farmer。
- PPO 的 `WP_ET_kg_m3` 在 036_01 输出中缺失，因此本任务不伪造 WP_ET；若要比较 WP_ET，需要后续用 checkpoint 做只评估重放并保存 DSSAT Summary/ET。
- FQA2018 的 PPO 产量为 0 已列入高优先级异常。
- 快速抽查 FQA2018 日值表显示：`topwt` 最高约 4213–4221 kg/ha，`grnwt` 始终为 0，episode 约 83 天结束；因此它更像是籽粒形成/物候终止异常或真实失败，而不是 PPO 动作没有进入 DSSAT。

## 汇总表

| station_code | checkpoint_step | validation_years | mean_final_grnwt | mean_total_irrigation | mean_total_n | mean_PFP_N | mean_WP_ET_kg_m3 | baseline_matched_years | any_metric_win_available_count | yield_win_available_count | wp_et_win_available_count | pfp_n_win_available_count | mean_gap_yield_vs_available_max | mean_gap_wp_et_vs_available_max | mean_gap_pfp_n_vs_available_max | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 25000 | 10 | 7188.6668 | 45.0 | 160.0 | 44.9292 |  | 10 | 1 | 1 | 0 | 0 | -249.3332 |  | -5.1056 | 0.6055 | 0.0122 |
| FQA | 50000 | 10 | 7205.1342 | 75.0 | 240.0 | 30.0214 |  | 10 | 1 | 1 | 0 | 0 | -232.8658 |  | -20.0133 | 0.5572 | 0.0122 |
| FQA | 75000 | 10 | 7187.1844 | 45.0 | 240.0 | 29.9466 |  | 10 | 1 | 1 | 0 | 0 | -250.8156 |  | -20.0881 | 0.6061 | 0.0122 |
| FQA | 100000 | 10 | 7187.1844 | 45.0 | 240.0 | 29.9466 |  | 10 | 1 | 1 | 0 | 0 | -250.8156 |  | -20.0881 | 0.6061 | 0.0122 |
| HLA | 25000 | 10 | 6795.0646 | 150.0 | 240.0 | 28.3128 |  | 10 | 7 | 7 | 0 | 0 | 302.1357 |  | -6.8772 | 0.0 | 0.0284 |
| HLA | 50000 | 10 | 6794.1802 | 135.0 | 240.0 | 28.3091 |  | 10 | 7 | 7 | 0 | 0 | 301.2513 |  | -6.8809 | 0.0 | 0.0301 |
| HLA | 75000 | 10 | 6794.0487 | 150.0 | 240.0 | 28.3085 |  | 10 | 7 | 7 | 0 | 0 | 301.1198 |  | -6.8815 | 0.0 | 0.0271 |
| HLA | 100000 | 10 | 6794.1917 | 150.0 | 240.0 | 28.3091 |  | 10 | 7 | 7 | 0 | 0 | 301.2627 |  | -6.8809 | 0.0 | 0.0309 |
| LCA | 25000 | 10 | 9317.1009 | 45.0 | 120.0 | 77.6425 |  | 10 | 10 | 7 | 0 | 10 | -43.0341 |  | 39.9025 | 0.0 | 0.0122 |
| LCA | 50000 | 10 | 9357.0422 | 150.0 | 200.0 | 46.7852 |  | 10 | 10 | 7 | 0 | 10 | -3.0927 |  | 9.0452 | 0.0 | 0.0122 |
| LCA | 75000 | 10 | 9316.4586 | 66.0 | 40.0 | 232.9115 |  | 10 | 10 | 7 | 0 | 10 | -43.6763 |  | 195.1715 | 0.0 | 0.0122 |
| LCA | 100000 | 10 | 9317.1994 | 45.0 | 40.0 | 232.93 |  | 10 | 10 | 7 | 0 | 10 | -42.9355 |  | 195.19 | 0.0 | 0.0122 |
| SYA | 25000 | 10 | 10141.3263 | 150.0 | 212.0 | 48.0679 |  | 10 | 10 | 5 | 0 | 10 | -14.2737 |  | 8.4679 | 0.6165 | 0.0437 |
| SYA | 50000 | 10 | 10160.0826 | 150.0 | 240.0 | 42.3337 |  | 10 | 10 | 4 | 0 | 10 | 4.4826 |  | 2.7337 | 0.5304 | 0.017 |
| SYA | 75000 | 10 | 9771.7183 | 136.5 | 160.0 | 61.0732 |  | 10 | 10 | 2 | 0 | 10 | -383.8817 |  | 21.4732 | 0.9601 | 0.3919 |
| SYA | 100000 | 10 | 9977.6743 | 135.0 | 240.0 | 41.5736 |  | 10 | 10 | 2 | 0 | 10 | -177.9257 |  | 1.9736 | 0.8779 | 0.017 |
| YCA | 25000 | 10 | 8156.3245 | 150.0 | 240.0 | 33.9847 |  | 10 | 10 | 0 | 0 | 10 | -47.2064 |  | 0.8921 | 0.073 | 0.0147 |
| YCA | 50000 | 10 | 8155.5513 | 135.0 | 80.0 | 101.9444 |  | 10 | 10 | 1 | 0 | 10 | -47.9796 |  | 68.8518 | 0.1785 | 0.2553 |
| YCA | 75000 | 10 | 8159.0017 | 150.0 | 240.0 | 33.9958 |  | 10 | 10 | 1 | 0 | 10 | -44.5292 |  | 0.9033 | 0.1614 | 0.0147 |
| YCA | 100000 | 10 | 8202.0087 | 45.0 | 80.0 | 102.5251 |  | 10 | 10 | 2 | 0 | 10 | -1.5222 |  | 69.4326 | 0.0 | 0.1341 |

## 异常与边界

| severity | issue | station_code | year | checkpoint_step | details |
| --- | --- | --- | --- | --- | --- |
| high | ppo_final_grnwt_zero | FQA | 2018 | 25000 | final_grnwt=0; daily_csv_path=benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun/daily_outputs/FQA/FQA_2018_seed0_ckpt25000_daily.csv |
| high | ppo_final_grnwt_zero | FQA | 2018 | 50000 | final_grnwt=0; daily_csv_path=benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun/daily_outputs/FQA/FQA_2018_seed0_ckpt50000_daily.csv |
| high | ppo_final_grnwt_zero | FQA | 2018 | 75000 | final_grnwt=0; daily_csv_path=benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun/daily_outputs/FQA/FQA_2018_seed0_ckpt75000_daily.csv |
| high | ppo_final_grnwt_zero | FQA | 2018 | 100000 | final_grnwt=0; daily_csv_path=benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun/daily_outputs/FQA/FQA_2018_seed0_ckpt100000_daily.csv |
| medium | ppo_wp_et_missing_all_rows | ALL |  |  | 200/200 PPO rows lack WP_ET_kg_m3; requires checkpoint replay with DSSAT summary retention. |
| low | multiple_recorded_family_rows | ALL |  |  | 94 station-years have more than one recorded_farmer/template row; available-baseline max is conservative but not always a single canonical recorded_farmer. |

## 基线覆盖概览

| station_code | years | min_group_count | max_group_count | multi_recorded_template_years |
| --- | --- | --- | --- | --- |
| FQA | 24 | 3 | 4 | 24 |
| HLA | 20 | 4 | 4 | 15 |
| LCA | 19 | 4 | 4 | 18 |
| SYA | 19 | 4 | 4 | 19 |
| YCA | 20 | 4 | 4 | 18 |

## 输出文件

- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_corrected_checkpoint_validation_summary.csv`
- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_corrected_by_station_checkpoint.csv`
- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_baseline_coverage_by_station_year.csv`
- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_anomaly_flags.csv`