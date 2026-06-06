# Action design debug report

Generated at: 2026-06-06

## Goal

This stage revises action design after reward/cost/terminal-profit tuning all failed to avoid 300/450 cap saturation.

## Implemented wrappers

- `ScheduledActionDesignWrapper`: shared implementation for decision interval and window-gated action filtering.
- `DecisionIntervalActionWrapper`: low-frequency decision wrapper.
- `PhenologyWindowActionWrapper`: phenology window wrapper.

## Wrapper order

`GymDssatWrapper -> ActionDesignWrapper -> SafeActionWrapper -> EpisodeProfitRewardWrapper`

This means PPO raw actions are filtered by the action design first, action safety remains the final physical guard, and terminal profit reward is computed after the environment step.

## HLA comparison

| action_design | action_design_type | eval_count | ok_count | mean_yield | std_yield | mean_reward | mean_irrigation | mean_n | mean_irrigation_saturation_ratio | mean_n_saturation_ratio | mean_swfac | mean_nstres | mean_design_filtered_days | mean_safety_trigger_days | mean_decision_days | yield_loss_vs_baseline | input_reduction_vs_baseline | strict_pass | relaxed_pass | action_too_restrictive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A10_decision_interval_10d | decision_interval | 3 | 3 | 7295.5361 | 474.9832 | -3.4114 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0006 | 10.3333 | 159.0 | 21.0 | -0.0607 | 0.0 | False | False | False |
| A10_plus_B_window_gated | decision_interval_plus_window | 3 | 3 | 6684.4651 | 255.6479 | -3.4308 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.1204 | 15.0 | 159.0 | 21.0 | 0.0282 | 0.0 | False | False | False |
| A15_decision_interval_15d | decision_interval | 3 | 3 | 6651.8732 | 374.0926 | -3.4317 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.1263 | 15.0 | 159.0 | 15.6667 | 0.0329 | 0.0 | False | False | False |
| A7_decision_interval_7d | decision_interval | 3 | 3 | 1919.2241 | 182.5435 | -3.5814 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.2558 | 5.0 | 159.0 | 27.3333 | 0.721 | 0.0 | False | False | False |
| B_window_gated_default | phenology_window | 3 | 3 | 6118.8019 | 445.5332 | -3.4485 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.1459 | 9.0 | 159.0 | 159.0 | 0.1104 | 0.0 | False | False | False |
| baseline_daily_action_current | baseline | 3 | 3 | 6878.2487 | 261.2437 | -3.4245 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0 | 159.0 | 159.0 | 0.0 | 0.0 | False | False | False |

## Pass summary

- Designs with both irrigation and N unsaturated: None
- Designs with yield loss <= 15%: A10_decision_interval_10d, A10_plus_B_window_gated, A15_decision_interval_15d, B_window_gated_default, baseline_daily_action_current
- Recommended action design: `None`

## SYA/LCA extension

| selected_action_design | run_status | notes |
| --- | --- | --- |
|  | skipped | No HLA action design passed strict or relaxed criteria. |

## Interpretation

If A/B wrappers are still saturated, the next step should be explicit seasonal budget action or scheduled discrete event actions. If A/B wrappers reduce input but collapse yield, the windows or stage budgets need agronomic calibration before multi-seed.

## Next step

Proceed to multi-seed only if the selected HLA action design is unsaturated and has yield loss <= 15%. Proceed to rainfall-scaling only after action behavior is interpretable in observed-year HLA.
