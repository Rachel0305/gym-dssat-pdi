# 041_05 SYA lowIC teacher warm-start 漂移审计记录

## 结论

- 本任务不训练、不重新运行 DSSAT，只读取 041_04 已有评估与日值表。
- 目标是解释：为什么 041_04 的 PPO fine-tune 相比 BC init 产量下降、PFP_N 上升。
- 输出分支：`A_drift_audit_completed`

## 主要数字

- 100K 相比 BC init，平均产量变化：-365.8 kg/ha。
- 100K 相比 BC init，平均 WP_ET 变化：-0.059 kg/m3。
- 100K 相比 BC init，平均 PFP_N 变化：6.480 kg/kg。
- 100K 相比 BC init，平均灌溉变化：-15.0 mm。
- 100K 相比 BC init，平均施氮变化：-40.0 kg/ha。

## 每年指标变化：PPO 100K - BC init

| year | delta_grain_yield_kg_ha | delta_WP_ET_kg_m3 | delta_PFP_N_kg_kg | delta_summary_irrigation_total | delta_summary_nitrogen_total | ppo100k_all3_win_vs_four_max |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | -1116.0 | -0.17 | 3.0 | -15.0 | -40.0 | False |
| 2015 | -516.0 | -0.1 | 6.6 | -15.0 | -40.0 | False |
| 2016 | 12.0 | 0.0 | 6.6 | -15.0 | -40.0 | True |
| 2017 | -545.0 | -0.05 | 5.5 | -15.0 | -40.0 | False |
| 2018 | 25.0 | 0.01 | 7.0 | -15.0 | -40.0 | True |
| 2019 | -381.0 | -0.07 | 6.8 | -15.0 | -40.0 | False |
| 2020 | -250.0 | -0.04 | 7.0 | -15.0 | -40.0 | False |
| 2021 | 31.0 | 0.01 | 8.6 | -15.0 | -40.0 | True |
| 2022 | -411.0 | -0.08 | 7.1 | -15.0 | -40.0 | False |
| 2023 | -507.0 | -0.1 | 6.6 | -15.0 | -40.0 | False |

## 每年管理总量与阶段分配

| year | policy_label | irrigation_total | nitrogen_total | irrigation_event_count | nitrogen_event_count | first_irrigation_day | last_irrigation_day | first_nitrogen_day | last_nitrogen_day | max_swfac | max_nstres | irrigation_D1_30 | nitrogen_D1_30 | irrigation_D31_60 | nitrogen_D31_60 | irrigation_D61_90 | nitrogen_D61_90 | irrigation_D91_plus | nitrogen_D91_plus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.849 | 0.11 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2014 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.905 | 0.221 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2015 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.088 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2015 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.0 | 0.394 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2016 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.015 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2016 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.0 | 0.015 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2017 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.827 | 0.014 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2017 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.855 | 0.151 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2018 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.017 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2018 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.0 | 0.017 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2019 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.013 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2019 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.0 | 0.162 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2020 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.016 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2020 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.496 | 0.156 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2021 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.014 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2021 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.0 | 0.023 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2022 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.19 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2022 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.0 | 0.431 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2023 | bc_init | 240.0 | 240.0 | 6 | 3 | 0.0 | 92.0 | 0.0 | 16.0 | 0.0 | 0.016 | 75.0 | 240.0 | 75.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |
| 2023 | ppo100k | 225.0 | 200.0 | 6 | 2 | 0.0 | 92.0 | 0.0 | 9.0 | 0.0 | 0.295 | 75.0 | 200.0 | 60.0 | 0.0 | 45.0 | 0.0 | 45.0 | 0.0 |

## 输出文件

- metric_delta: `benchmark_results/041_05_sya_lowIC_teacher_warmstart_drift_audit/tables/041_05_metric_delta_ppo100k_minus_bcinit.csv`
- event_summary: `benchmark_results/041_05_sya_lowIC_teacher_warmstart_drift_audit/tables/041_05_event_stage_summary.csv`
- events: `benchmark_results/041_05_sya_lowIC_teacher_warmstart_drift_audit/tables/041_05_management_events_bcinit_vs_ppo100k.csv`
- figures: `benchmark_results/041_05_sya_lowIC_teacher_warmstart_drift_audit/figures/041_05_metric_drift_ppo100k_minus_bcinit.png`
- figures: `benchmark_results/041_05_sya_lowIC_teacher_warmstart_drift_audit/figures/041_05_resource_totals_bcinit_vs_ppo100k.png`
- figures: `benchmark_results/041_05_sya_lowIC_teacher_warmstart_drift_audit/figures/041_05_irrigation_stage_allocation.png`
- figures: `benchmark_results/041_05_sya_lowIC_teacher_warmstart_drift_audit/figures/041_05_nitrogen_stage_allocation.png`
- record_md: `docs/041_05_sya_lowIC_teacher_warmstart_drift_audit_record.md`
