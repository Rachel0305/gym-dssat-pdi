# Prior replay debug report

Generated at: 2026-06-06

## Mismatch cause

006_10 FT0 did not reproduce 006_09 because it used a different action safety interface:

1. `SafeActionWrapper` read environment `dap`, which is 0 on early rows; first-day N events were rejected by `anfer_dap_range`.
2. 006_10 used `daily_n_max = 80`, clipping HLA/LCA BC prior N events above 80 kg/ha.
3. 006_09 replay used exported BC action tables with sim-day action safety and daily N max 150.

## Pure replay result

| index | final_grnwt | total_n_fertilizer | total_irrigation | yield_loss_vs_00609_bc_two_stage | n_diff_vs_00609_bc_two_stage |
| --- | --- | --- | --- | --- | --- |
| mean_yield | 8091.8537 |  |  |  |  |
| mean_n |  | 153.575 |  |  |  |
| mean_i |  |  | 0.0 |  |  |
| mean_yield_loss |  |  |  | -0.0 |  |
| mean_n_diff |  |  |  |  | -0.0 |

## Fixed constrained PPO result

| finetune_method | eval_count | mean_yield | mean_profit | mean_i | mean_n | mean_loss_ppo | mean_loss_bc | profit_gap_vs_bc_two_stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT0_BC_two_stage_replay_fixed | 10 | 8091.8537 | 42.5248 | 0.0 | 153.575 | -0.1768 | -0.0 | 0.0 |
| FT2_residual_bc_prior_fixed | 10 | 6874.2549 | -1.1707 | 71.192 | 137.2691 | 0.0002 | 0.2089 | -43.6955 |
| FT3_budget_guardrail_100_200_fixed | 10 | 8460.3945 | -14.8825 | 98.973 | 200.0 | -0.2304 | -0.0453 | -57.4073 |

## Recommendation

| finetune_method | eval_count | mean_yield | mean_profit | mean_i | mean_n | mean_loss_ppo | mean_loss_bc | profit_gap_vs_bc_two_stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT0_BC_two_stage_replay_fixed | 10 | 8091.8537 | 42.5248 | 0.0 | 153.575 | -0.1768 | -0.0 | 0.0 |
