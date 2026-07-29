# 040_06 SYA lowIC PPO 灌溉预算分配审计记录

## 结论先说

本任务固定 PPO checkpoint 75000，只读取已有 daily CSV，不重新训练、不重跑 DSSAT。

核心发现：PPO 在 2014–2023 验证年中几乎把全部灌溉预算集中在前 30 DAP。  
这支持 040_05 的机制判断：2014/2017 的水分胁迫失败不是因为完全没有水，而是因为水用得过早，后期胁迫出现后没有剩余灌溉响应。

## 逐年灌溉分配

| year | diagnosis_class | final_grnwt | ppo_gap_yield_vs_expert | total_irrigation | irrigation_DAP1_30_mm | irrigation_DAP1_30_share | irrigation_DAP31_60_mm | irrigation_DAP61_90_mm | irrigation_DAP91_plus_mm | first_swfac_gt_0p05_dap | swfac_days_gt_0p05 | irrigation_after_first_stress_mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | water_stress_failure | 5506.1548 | -5334.1919 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 86.0 | 27 | 0.0 |
| 2015 | yield_gap_other | 7951.8073 | -2906.0602 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 90.0 | 10 | 0.0 |
| 2016 | yield_win | 7998.5632 | 126.7847 | 150.0 | 135.0 | 0.9 | 0.0 | 15.0 | 0.0 |  | 0 | 0.0 |
| 2017 | water_stress_failure | 5204.1901 | -5691.8475 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 68.0 | 31 | 0.0 |
| 2018 | yield_gap_other | 7321.0339 | -773.1421 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 101.0 | 8 | 0.0 |
| 2019 | yield_gap_other | 8554.1412 | -1810.3925 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 79.0 | 8 | 0.0 |
| 2020 | yield_gap_other | 6629.4525 | -3241.2402 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 82.0 | 18 | 0.0 |
| 2021 | yield_win | 10437.9456 | 357.9706 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 |  | 0 | 0.0 |
| 2022 | near_yield_resource_saving | 10898.3008 | -24.1479 | 150.0 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 |  | 0 | 0.0 |
| 2023 | yield_gap_other | 10075.7513 | -861.0138 | 150.0 | 135.0 | 0.9 | 0.0 | 15.0 | 0.0 | 109.0 | 4 | 0.0 |

## 按失败类型汇总

| diagnosis_class | year_count | mean_yield_gap_vs_expert | mean_total_irrigation | mean_swfac_days_gt_0p05 | mean_first_swfac_gt_0p05_dap | mean_irrigation_DAP1_30_share | mean_irrigation_DAP31_60_share | mean_irrigation_DAP61_90_share | mean_irrigation_DAP91_plus_share | years_with_irrigation_after_first_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yield_gap_other | 5 | -1918.3698 | 150.0 | 9.6 | 92.2 | 0.98 | 0.0 | 0.02 | 0.0 | 0 |
| water_stress_failure | 2 | -5513.0197 | 150.0 | 29.0 | 77.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0 |
| yield_win | 2 | 242.3776 | 150.0 | 0.0 |  | 0.95 | 0.0 | 0.05 | 0.0 | 0 |
| near_yield_resource_saving | 1 | -24.1479 | 150.0 | 0.0 |  | 1.0 | 0.0 | 0.0 | 0.0 | 0 |

## 解释

- 如果 `irrigation_DAP1_30_share` 接近 1，说明 PPO 几乎把水全用在前 30 天。
- 如果 `first_swfac_gt_0p05_dap` 远晚于最后一次灌溉 DAP，且 `irrigation_after_first_stress_mm = 0`，说明 PPO 没有为后期水分胁迫保留预算。
- 这一步仍是审计，不是改 reward 或改约束。

## 输出文件

- 逐年表：`benchmark_results\040_06_sya_lowIC_ppo_irrigation_budget_distribution_audit\tables\040_06_irrigation_distribution_by_year.csv`
- 类型汇总表：`benchmark_results\040_06_sya_lowIC_ppo_irrigation_budget_distribution_audit\tables\040_06_irrigation_distribution_by_diagnosis_class.csv`

