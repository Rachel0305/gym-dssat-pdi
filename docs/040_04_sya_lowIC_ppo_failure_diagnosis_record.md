# 040_04 SYA lowIC PPO 失败年份诊断记录

## 结论先说

本任务回到 PPO 主线，固定使用 040_03 选出的 PPO 最佳 checkpoint：**75000**。  
诊断对象是 SYA lowIC 验证年份 2014–2023。

这一步没有重新训练，也没有重新运行 DSSAT；只读取已有 PPO 评估结果和 039_02 lowIC 三基线结果。

## 逐年诊断表

| year | ppo_yield | expert_yield | ppo_gap_yield_vs_expert | ppo_water_saved_vs_expert | ppo_n_saved_vs_expert | ppo_swfac_days_gt_0p05 | ppo_nstres_days_gt_0p05 | diagnosis_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 5506.15 | 10840.35 | -5334.19 | 116.00 | 57.00 | 27 | 0 | water_stress_failure |
| 2015 | 7951.81 | 10857.87 | -2906.06 | 116.00 | 57.00 | 10 | 0 | yield_gap_other |
| 2016 | 7998.56 | 7871.78 | 126.78 | 116.00 | 57.00 | 0 | 0 | yield_win |
| 2017 | 5204.19 | 10896.04 | -5691.85 | 116.00 | 57.00 | 31 | 0 | water_stress_failure |
| 2018 | 7321.03 | 8094.18 | -773.14 | 116.00 | 57.00 | 8 | 0 | yield_gap_other |
| 2019 | 8554.14 | 10364.53 | -1810.39 | 116.00 | 57.00 | 8 | 0 | yield_gap_other |
| 2020 | 6629.45 | 9870.69 | -3241.24 | 116.00 | 57.00 | 18 | 0 | yield_gap_other |
| 2021 | 10437.95 | 10079.97 | 357.97 | 116.00 | 57.00 | 0 | 0 | yield_win |
| 2022 | 10898.30 | 10922.45 | -24.15 | 116.00 | 57.00 | 0 | 0 | near_yield_resource_saving |
| 2023 | 10075.75 | 10936.77 | -861.01 | 116.00 | 57.00 | 4 | 0 | yield_gap_other |

## 类型汇总

| diagnosis_class | year_count | mean_yield_gap_vs_expert | mean_water_saved_vs_expert | mean_n_saved_vs_expert | mean_swfac_days |
| --- | --- | --- | --- | --- | --- |
| yield_gap_other | 5 | -1918.37 | 116.00 | 57.00 | 9.60 |
| water_stress_failure | 2 | -5513.02 | 116.00 | 57.00 | 29.00 |
| yield_win | 2 | 242.38 | 116.00 | 57.00 | 0.00 |
| near_yield_resource_saving | 1 | -24.15 | 116.00 | 57.00 | 0.00 |

## 解释

- `water_stress_failure`：PPO 相比 expert 产量明显偏低，同时水分胁迫天数较多，说明下一步 PPO 优化应优先处理水分胁迫压不住的问题。
- `near_yield_resource_saving`：产量接近 expert，同时节水或节氮，属于有希望通过轻量 guardrail 或 checkpoint 选择改善的年份。
- `yield_win`：PPO 产量超过三基线最高值。

## 输出文件

- 逐年诊断表：`benchmark_results\040_04_sya_lowIC_ppo_failure_diagnosis\tables\040_04_ppo_vs_lowIC_baselines_by_year.csv`
- 类型汇总表：`benchmark_results\040_04_sya_lowIC_ppo_failure_diagnosis\tables\040_04_ppo_failure_type_summary.csv`

