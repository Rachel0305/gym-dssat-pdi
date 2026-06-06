# Action design candidate plan

| action_design | type | decision_interval_days | fertilization_windows | irrigation_windows | reward_version | cap |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_daily_action_current | baseline |  |  |  | P6_high_economic_pressure | 300 mm irrigation / 450 kg ha-1 N |
| A7_decision_interval_7d | decision_interval | 7 |  |  | P6_high_economic_pressure | 300 mm irrigation / 450 kg ha-1 N |
| A10_decision_interval_10d | decision_interval | 10 |  |  | P6_high_economic_pressure | 300 mm irrigation / 450 kg ha-1 N |
| A15_decision_interval_15d | decision_interval | 15 |  |  | P6_high_economic_pressure | 300 mm irrigation / 450 kg ha-1 N |
| B_window_gated_default | phenology_window |  | [[1, 7], [25, 40], [55, 70]] | [[20, 35], [45, 65], [70, 95]] | P6_high_economic_pressure | 300 mm irrigation / 450 kg ha-1 N |
| A10_plus_B_window_gated | decision_interval_plus_window | 10 | [[1, 7], [25, 40], [55, 70]] | [[20, 35], [45, 65], [70, 95]] | P6_high_economic_pressure | 300 mm irrigation / 450 kg ha-1 N |

Design C, explicit seasonal budget allocation, is documented as the next option if A/B wrappers still fail.