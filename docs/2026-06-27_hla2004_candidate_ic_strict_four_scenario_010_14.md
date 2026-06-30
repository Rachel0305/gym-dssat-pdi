# 2026-06-27 HLA 2004 候选 IC 严格四情景 forward 对照

## 运行性质

- 本轮不训练 PPO，只做 4 个同输入条件下的 forward simulation。
- 四个情景全部使用 HLA 2004 候选 IC：`0.55 + 0.25N`。
- recorded expert 和 PPO 是旧 008_19 动作表在候选 IC 下的回放，不是重新优化。

## 结果表

| scenario_label | grain_yield_gwad | biomass_cwad | total_irrigation_from_actions_or_events | total_n_from_actions_or_events | summary_ircm | summary_nicm | max_wspd | max_nstd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate IC null | 0.000 | 3478.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.391 |
| Recorded expert replay | 242.000 | 5698.000 | 30.000 | 165.000 | 30.000 | 165.000 | 1.000 | 0.018 |
| DSSAT auto irrigation + auto-N attempt | 2501.000 | 8287.000 | 329.000 | 0.000 | 329.000 | 0.000 | 0.000 | 0.594 |
| PPO action replay | 1552.000 | 6611.000 | 80.000 | 150.000 | 80.000 | 150.000 | 1.000 | 0.357 |

## DSSAT 自动管理触发情况

- 自动灌溉事件数：7，合计 329.00 mm。
- 自动施肥事件数：0，合计 0.00。

## 初步结论

- DSSAT 自动灌溉成功触发，但自动施肥仍未触发，因此第 3 情景不能称为完整自动水氮管理。

## 输出文件

- `hla2004_candidate_ic_strict_four_scenario_daily_values.csv`
- `hla2004_candidate_ic_strict_four_scenario_summary.csv`
- `hla2004_candidate_ic_strict_four_scenario_management_events.csv`
- `hla2004_candidate_ic_strict_four_scenario_process.png`
- `hla2004_candidate_ic_strict_four_scenario_final_yield_biomass.png`
