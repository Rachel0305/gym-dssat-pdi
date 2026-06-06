# Episode-level profit reward debug report

Generated at: 2026-06-06

## Goal

This stage redesigns the reward as an episode-level profit objective under action-safe PPO. It does not modify the original site-packages reward, does not train PPO without action safety, and does not overwrite 006_03/006_04 results.

## Why step-wise cost tuning failed

006_03 and 006_04 both changed daily cost terms, but PPO still saturated the 300 mm irrigation / 450 kg ha-1 N cap. That suggests the previous reward signal did not provide a clear season-level trade-off between final yield and total input.

## Terminal reward feasibility

Terminal reward is feasible at the wrapper layer. The original reward callback lacks a done flag, so this stage uses `EpisodeProfitRewardWrapper` around the action-safe env. The wrapper reads final `grnwt` after the environment returns done and accumulates safe `amir`/`anfer` from `SafeActionWrapper`.

## Reward candidates tested

P0_current_reward_baseline, P1_terminal_profit_weak_cost, P2_terminal_profit_medium_cost, P3_terminal_profit_strong_cost, P4_terminal_profit_with_daily_cost, P5_lower_grain_value_medium_cost, P6_high_economic_pressure

## HLA comparison

| reward_version | reward_family | eval_count | ok_count | mean_yield | std_yield | mean_reward | mean_irrigation | mean_n | mean_irrigation_saturation_ratio | mean_n_saturation_ratio | profit_score | mean_swfac | mean_nstres | yield_loss_vs_baseline | input_reduction_vs_baseline | strict_pass | relaxed_pass | cost_too_strong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0_current_reward_baseline | P0_baseline | 3 | 3 | 6843.574 | 273.1875 | 66.8409 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0007 | 0.0 | 0.0 | False | False | False |
| P1_terminal_profit_weak_cost | P_terminal_profit | 3 | 3 | 6958.8615 | 265.896 | 0.1084 | 300.0 | 450.0 | 1.0 | 1.0 | 17.0886 | 0.0 | 0.0007 | -0.0168 | 0.0 | False | False | False |
| P2_terminal_profit_medium_cost | P_terminal_profit | 3 | 3 | 6870.1542 | 264.0959 | -1.2214 | 300.0 | 450.0 | 1.0 | 1.0 | -193.7985 | 0.0 | 0.0007 | -0.0039 | 0.0 | False | False | False |
| P3_terminal_profit_strong_cost | P_terminal_profit | 3 | 3 | 6869.8389 | 262.9457 | -2.8767 | 300.0 | 450.0 | 1.0 | 1.0 | -456.3016 | 0.0 | 0.0007 | -0.0038 | 0.0 | False | False | False |
| P4_terminal_profit_with_daily_cost | P_terminal_plus_daily_cost | 3 | 3 | 6878.2011 | 261.6797 | -1.3722 | 300.0 | 450.0 | 1.0 | 1.0 | -217.718 | 0.0 | 0.0007 | -0.0051 | 0.0 | False | False | False |
| P5_lower_grain_value_medium_cost | P_terminal_plus_daily_cost | 3 | 3 | 6874.0544 | 263.3896 | -1.5896 | 300.0 | 450.0 | 1.0 | 1.0 | -252.1297 | 0.0 | 0.0007 | -0.0045 | 0.0 | False | False | False |
| P6_high_economic_pressure | P_terminal_plus_daily_cost | 3 | 3 | 6878.2487 | 261.2437 | -3.4245 | 300.0 | 450.0 | 1.0 | 1.0 | -543.1088 | 0.0 | 0.0007 | -0.0051 | 0.0 | False | False | False |

## Pass summary

- Candidates with both irrigation and N unsaturated: None
- Candidates with yield loss <= 15%: P0_current_reward_baseline, P1_terminal_profit_weak_cost, P2_terminal_profit_medium_cost, P3_terminal_profit_strong_cost, P4_terminal_profit_with_daily_cost, P5_lower_grain_value_medium_cost, P6_high_economic_pressure
- Recommended candidate: `None`

## SYA/LCA extension

| selected_reward | run_status | notes |
| --- | --- | --- |
|  | skipped | No HLA candidate passed strict or relaxed criteria. |

## Interpretation

If all P candidates still hit the cap, the remaining problem is probably not just coefficient size. The next likely direction is action-design revision, such as scheduled discrete management events, lower-frequency decisions, or explicit season budget actions, before multi-seed or rainfall-scaling expansion.

## Next step

Proceed to multi-seed only if a candidate is unsaturated and has acceptable yield loss. Proceed to rainfall-scaling budget scenario only after the action/reward behavior is interpretable in the observed-year HLA pilot.
