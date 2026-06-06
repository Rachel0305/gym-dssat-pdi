# A/B action design failure review

Generated at: 2026-06-06

The 006_06 decision-interval and window-gated wrappers reduced action opportunities, but did not change the total-budget decision. PPO could still request large positive actions on the remaining allowed days, and SafeActionWrapper still spent the entire 300/450 seasonal cap.

The high safety-trigger-day counts show that action safety remained the real cap controller. A7/A10/A15 and B window gating filtered daily actions but did not make PPO choose a smaller seasonal budget.

Therefore 006_07 moves from daily action filtering to explicit budget or event semantics: PPO selects a season budget, scheduled event amounts, or stage budgets; wrapper rules execute those decisions.


## 006_06 comparison

| action_design | mean_irrigation | mean_n | mean_irrigation_saturation_ratio | mean_n_saturation_ratio | mean_design_filtered_days | mean_safety_trigger_days | mean_decision_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A10_decision_interval_10d | 300.0 | 450.0 | 1.0 | 1.0 | 10.333333333333334 | 159.0 | 21.0 |
| A10_plus_B_window_gated | 300.0 | 450.0 | 1.0 | 1.0 | 15.0 | 159.0 | 21.0 |
| A15_decision_interval_15d | 300.0 | 450.0 | 1.0 | 1.0 | 15.0 | 159.0 | 15.666666666666666 |
| A7_decision_interval_7d | 300.0 | 450.0 | 1.0 | 1.0 | 5.0 | 159.0 | 27.33333333333333 |
| B_window_gated_default | 300.0 | 450.0 | 1.0 | 1.0 | 9.0 | 159.0 | 159.0 |
| baseline_daily_action_current | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 159.0 | 159.0 |