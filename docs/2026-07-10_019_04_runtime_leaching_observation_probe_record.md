# 019_04 runtime leaching observation probe 记录

## 结论先行

本轮使用 FQ2016 两个确定性回放情景，不训练，只验证 gym/PDI 运行时 observation/full state 是否能读到 leaching 相关变量。

结论：`observation_has_cleach=False`，但 `full_state_has_cleach=True`。这说明 `cleach/tleachd/cnox` 已经由 PDI 传到 gym 的 full state，只是当前 `latest_observation_dict()` 和主线日值表没有读取 `_history['state']`，所以普通 observation 里看不到。

关键对照：`I120_N300` 中 `full_state_final_cleach=2.717`，`SoilNi.OUT` 的 `NLCC=2.720`，两者基本一致。因此当前不需要先改 DSSAT/PDI 模板；下一步应该先把 full state 中的 `cleach/tleachd/cnox` 写入评估日值表，并在 reward wrapper 中从 full state 读取。

## 结果

| case | planned_irrigation | planned_nitrogen | observation_has_cleach | full_state_has_cleach | runtime_final_cleach | full_state_final_cleach | runtime_sum_tleachd | full_state_sum_tleachd | summary_final_NLCM | soilni_final_NLCC | plantgro_final_GWAD | harvest_yield |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I0_N0 | 0.000 | 0.000 | 0.000 | 1.000 |  | 0.000 | 0.000 | 0.000 | 163.000 | 0.000 | 7066.000 | 7066.000 |
| I120_N300 | 120.000 | 300.000 | 0.000 | 1.000 |  | 2.717 | 0.000 | 2.717 | 177.000 | 2.720 | 7970.000 | 7970.000 |

## 文件

- 汇总表：`DSSAT_auto_validation/runtime_leaching_observation_probe_019_04/019_04_runtime_leaching_summary.csv`
- 输出目录：`DSSAT_auto_validation/runtime_leaching_observation_probe_019_04`

## 下一步

本轮通过了 full-state 可用性验证。下一步做 leaching-aware reward smoke test：先只在一个站点年份上加入 `- leaching_cost * delta_cleach`，并保持其他奖励项不变。
