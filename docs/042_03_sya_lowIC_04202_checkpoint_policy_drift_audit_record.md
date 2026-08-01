# 042_03 SYA lowIC 042_02 checkpoint 策略漂移审计记录

## 任务性质

- 只读取 `042_02_rerun100k` 已有结果；不训练；不运行 DSSAT；不修改原始结果。
- 目的：解释为什么 75K checkpoint 表现较好，而 100K checkpoint 退化。

## 输入

- validation summary: `benchmark_results\042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k\evaluation\041_03_checkpoint_validation_summary.csv`
- by checkpoint: `benchmark_results\042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k\evaluation\041_03_validation_summary_by_checkpoint.csv`
- training inventory: `benchmark_results\042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k\evaluation\041_03_training_checkpoint_inventory.csv`

## checkpoint 汇总

| checkpoint_step | mean_yield | mean_wp_et | mean_pfp_n | mean_irrigation | mean_nitrogen | mean_irrigation_event_count | mean_nitrogen_event_count | any_metric_win_years | all3_win_years | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9955.0 | 2.104 | 41.49 | 240.0 | 240.0 | 6.0 | 3.0 | 10 | 2 | 0.8064 | 0.2593 |
| 25000 | 3847.4 | 0.893 |  | 225.0 | 0.0 | 5.0 | 0.0 | 0 | 0 | 0.0 | 0.556 |
| 50000 | 3847.4 | 0.893 |  | 225.0 | 0.0 | 5.0 | 0.0 | 0 | 0 | 0.0 | 0.556 |
| 75000 | 9931.7 | 2.145 | 41.36 | 225.0 | 240.0 | 5.0 | 3.0 | 10 | 6 | 0.8592 | 0.238 |
| 100000 | 3847.4 | 0.893 |  | 225.0 | 0.0 | 5.0 | 0.0 | 0 | 0 | 0.0 | 0.556 |

## 75K vs 100K 配对差异

| year | grain_yield_kg_ha_75k | grain_yield_kg_ha_100k | delta_yield_100k_minus_75k | summary_nitrogen_total_75k | summary_nitrogen_total_100k | delta_n_100k_minus_75k | nitrogen_lost_by_100k | all3_win_vs_four_max_75k | all3_win_vs_four_max_100k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 10208.0 | 4075.0 | -6133.0 | 240.0 | 0.0 | -240.0 | True | False | False |
| 2015 | 11095.0 | 3937.0 | -7158.0 | 240.0 | 0.0 | -240.0 | True | True | False |
| 2016 | 7876.0 | 3835.0 | -4041.0 | 240.0 | 0.0 | -240.0 | True | True | False |
| 2017 | 9896.0 | 2756.0 | -7140.0 | 240.0 | 0.0 | -240.0 | True | False | False |
| 2018 | 8293.0 | 3816.0 | -4477.0 | 240.0 | 0.0 | -240.0 | True | True | False |
| 2019 | 10427.0 | 3283.0 | -7144.0 | 240.0 | 0.0 | -240.0 | True | True | False |
| 2020 | 9945.0 | 4181.0 | -5764.0 | 240.0 | 0.0 | -240.0 | True | True | False |
| 2021 | 10114.0 | 4707.0 | -5407.0 | 240.0 | 0.0 | -240.0 | True | True | False |
| 2022 | 10732.0 | 5041.0 | -5691.0 | 240.0 | 0.0 | -240.0 | True | False | False |
| 2023 | 10731.0 | 2843.0 | -7888.0 | 240.0 | 0.0 | -240.0 | True | False | False |

## 判定

- 分支：`A_late_finetune_nitrogen_collapse_confirmed`
- 100K 相对 75K 丢失施氮的年份数：10/10。
- 如果本分支为 A，说明 100K 退化不是随机个别年份，而是后期 fine-tune 系统性把施氮策略推到 N=0。
- 本任务不能直接授权事后采用 75K；只能支持下一步预注册 checkpoint 选择或 early stopping 规则。

## 限制

- `042_02` 训练器没有写出训练年份 reset log，因此本任务不能审计训练年份采样偏差。
- 如需检查 reset 分布，下一轮训练脚本需显式保存 `RandomYearEnv.switch_log`。

## 输出

- action metrics: `benchmark_results\042_03_sya_lowIC_04202_checkpoint_policy_drift_audit\tables\042_03_checkpoint_year_action_metrics.csv`
- checkpoint summary: `benchmark_results\042_03_sya_lowIC_04202_checkpoint_policy_drift_audit\tables\042_03_checkpoint_drift_summary.csv`
- nitrogen events pivot: `benchmark_results\042_03_sya_lowIC_04202_checkpoint_policy_drift_audit\tables\042_03_nitrogen_events_by_year_checkpoint.csv`
- 75K vs 100K paired drift: `benchmark_results\042_03_sya_lowIC_04202_checkpoint_policy_drift_audit\tables\042_03_75k_vs_100k_policy_drift.csv`
