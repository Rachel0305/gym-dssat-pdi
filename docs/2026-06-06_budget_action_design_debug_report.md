# Budget/action design debug report

Generated at: 2026-06-06

## Goal

This stage moves beyond daily filtering to explicit seasonal budget, scheduled event, and stage budget action semantics. It does not modify my_data or site-packages reward files and does not overwrite 006_03/006_04/006_05/006_06.

## Implemented wrappers

- `SeasonalBudgetActionWrapper`: PPO chooses season water/N budget on DAP 1; wrapper releases fixed splits.
- `ScheduledEventActionWrapper`: PPO chooses event amounts only at fixed DAP events.
- `StageBudgetActionWrapper`: PPO chooses stage budget at stage starts.

## Wrapper order

`GymDssatWrapper -> BudgetActionWrapper -> SafeActionWrapper -> EpisodeProfitRewardWrapper`

## HLA comparison

| action_design | action_design_type | reward_version | eval_count | ok_count | mean_yield | std_yield | mean_reward | mean_irrigation | mean_n | mean_irrigation_saturation_ratio | mean_n_saturation_ratio | mean_swfac | mean_nstres | mean_budget_decision_days | mean_scheduled_event_days | mean_safe_trigger_days | yield_loss_vs_baseline | input_reduction_vs_baseline | strict_pass | relaxed_pass | action_too_restrictive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_plus_profit_reward | seasonal_budget | P6_high_economic_pressure | 3 | 3 | 7210.7456 | 552.9232 | -3.414 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0961 | 6.0 | 11.0 | 158.0 | -0.0487 | 0.0 | False | False | False |
| D_seasonal_budget_fixed_split | seasonal_budget | P0_current_reward_baseline | 3 | 3 | 7263.3217 | 517.3164 | 95.2137 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0251 | 6.0 | 11.0 | 158.0 | -0.0563 | 0.0 | False | False | False |
| E_plus_profit_reward | scheduled_event | P6_high_economic_pressure | 3 | 3 | 2131.5959 | 225.1673 | -3.5747 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.2489 | 0.0 | 11.0 | 158.0 | 0.69 | 0.0 | False | False | False |
| E_scheduled_events_default | scheduled_event | P0_current_reward_baseline | 3 | 3 | 2818.3242 | 446.3554 | 51.4736 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.2157 | 0.0 | 11.0 | 158.0 | 0.5901 | 0.0 | False | False | False |
| F_stage_budget_default | stage_budget | P0_current_reward_baseline | 3 | 3 | 6686.1351 | 301.3468 | 96.3501 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.1214 | 9.0 | 9.0 | 158.0 | 0.0276 | 0.0 | False | False | False |
| baseline_daily_action_current | baseline | P6_high_economic_pressure | 3 | 3 | 6875.8716 | 264.3272 | -3.4246 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0 | 0.0 | 158.0 | 0.0 | 0.0 | False | False | False |

## Pass summary

- Designs with both irrigation and N unsaturated: None
- Designs with yield loss <= 15%: D_plus_profit_reward, D_seasonal_budget_fixed_split, F_stage_budget_default, baseline_daily_action_current
- Recommended design: `None`

## SYA/LCA extension

| selected_action_design | run_status | notes |
| --- | --- | --- |
|  | skipped | No HLA budget/event design passed strict or relaxed criteria. |

## Interpretation

If D/E/F still select cap-level inputs, PPO is choosing maximum budget even when action semantics are explicit. If D/E/F reduce input but lose too much yield, the next step is offline schedule search or Bayesian optimization to build a stronger prior before RL.

## Next step

Do not enter multi-seed or rainfall-scaling until a budget/event design is interpretable. If this stage fails, shift to imitation learning, offline schedule search, Bayesian optimization, or DSSAT scenario ensemble.
