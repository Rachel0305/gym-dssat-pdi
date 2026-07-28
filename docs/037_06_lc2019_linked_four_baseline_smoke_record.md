# 037_06：LC2019 linked-action 四基线可信烟测记录

## 结论先说

- 成功情景数：4 / 4
- 手工基线链路通过数：1 / 2
- 失败数：0
- 耗时：26.6 秒

本任务验证一种可信基线重建方式：手工基线不再依赖静态 `fileX.MZX` 管理表，而是通过 linked management step action 按计划日期精确发送水肥动作。

## 管理事件链路审计

| scenario | manual_schedule | planned_i_events | mgmt_i_events | season_last_dap | post_harvest_planned_daps | planned_i_total | sent_i_total | mgmt_i_total | summary_i_total | planned_n_events | mgmt_n_events | planned_n_total | sent_n_total | mgmt_n_total | summary_n_total | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | False | 0 | 0 | 87 |  | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | ok_auto_or_null_recorded |
| recorded_farmer_template | True | 2 | 2 | 87 |  | 130.0 | 130.0 | 100.0 | 100.0 | 2 | 2 | 250.0 | 250.0 | 250.0 | 250.0 | event_chain_issue |
| official_extension_expert | True | 5 | 5 | 87 | 100 | 198.75 | 198.75 | 198.8 | 199.0 | 5 | 5 | 247.5 | 247.5 | 247.0 | 248.0 | ok_linked_manual_baseline |
| dssat_auto | False | 0 | 2 | 87 |  | 0.0 | 0.0 | 88.9 | 89.0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | ok_auto_or_null_recorded |

## 四情景结果

| scenario | grain_yield_kg_ha | summary_irrigation_mm | summary_n_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | max_water_stress | max_nitrogen_stress | planned_irrigation_event_count | planned_n_event_count | mgmt_irrigation_event_count | mgmt_n_event_count | snapshot_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 9807.0001 | 0.0 | 0.0 | 2.92 |  | 0.0 | 0.0122 | 0 | 0 | 0 | 0 | benchmark_results/037_06_lc2019_linked_four_baseline_smoke/snapshots/LCA/2019/null |
| recorded_farmer_template | 9807.0001 | 100.0 | 250.0 | 2.84 | 39.2 | 0.0 | 0.0122 | 2 | 2 | 2 | 2 | benchmark_results/037_06_lc2019_linked_four_baseline_smoke/snapshots/LCA/2019/recorded_farmer_template |
| official_extension_expert | 9807.0001 | 199.0 | 248.0 | 2.75 | 39.6 | 0.0 | 0.0122 | 5 | 5 | 5 | 5 | benchmark_results/037_06_lc2019_linked_four_baseline_smoke/snapshots/LCA/2019/official_extension_expert |
| dssat_auto | 9807.0001 | 89.0 | 0.0 | 2.77 |  | 0.0 | 0.0122 | 0 | 0 | 2 | 0 | benchmark_results/037_06_lc2019_linked_four_baseline_smoke/snapshots/LCA/2019/dssat_auto |

## 失败记录

无记录

## 输出文件

- summary：`benchmark_results/037_06_lc2019_linked_four_baseline_smoke/evaluation/037_06_lc2019_four_baseline_summary.csv`
- daily：`benchmark_results/037_06_lc2019_linked_four_baseline_smoke/evaluation/037_06_lc2019_four_baseline_daily.csv`
- audit：`benchmark_results/037_06_lc2019_linked_four_baseline_smoke/evaluation/037_06_lc2019_management_event_audit.csv`
