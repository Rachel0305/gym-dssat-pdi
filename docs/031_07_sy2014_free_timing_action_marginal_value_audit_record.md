# 031_07 SY2014 free-timing action marginal value audit record

## Scope

- SYA2014 only.
- No RL training.
- Deterministic DSSAT branches under a fixed `uniform_spread` / `expert_window_budget` background schedule.
- At each target DAP, only that day's action is replaced by one 3x3 candidate action.

## Background schedule

| DAP | irrigation | nitrogen |
|---:|---:|---:|
| 1 | 30.0 | 50.0 |
| 30 | 30.0 | 50.0 |
| 50 | 30.0 | 50.0 |
| 65 | 30.0 | 50.0 |
| 85 | 20.0 | 50.0 |
| 110 | 20.0 | 0.0 |

## Best action by target DAP, ranked by simple profit

| target_dap | candidate_i | candidate_n | safe_i_at_target | safe_n_at_target | final_grnwt | total_irrigation | total_n | simple_profit | delta_vs_noop_simple_profit | delta_vs_background_simple_profit | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10936.9775 | 130.0000 | 200.0000 | 9806.9775 | 0.0000 | 308.3740 | 0 | 11 |
| 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10914.4385 | 130.0000 | 200.0000 | 9784.4385 | 0.0000 | 285.8350 | 0 | 11 |
| 50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10946.9678 | 130.0000 | 200.0000 | 9816.9678 | 0.0000 | 318.3643 | 0 | 13 |
| 65 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10987.4512 | 130.0000 | 200.0000 | 9857.4512 | 0.0000 | 358.8477 | 0 | 13 |
| 85 | 20.0000 | 0.0000 | 20.0000 | 0.0000 | 10908.6035 | 160.0000 | 200.0000 | 9748.6035 | 160.3918 | 250.0000 | 0 | 14 |
| 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10908.6035 | 160.0000 | 250.0000 | 9498.6035 | 0.0000 | 0.0000 | 0 | 0 |

## Worst action by target DAP, ranked by simple profit

| target_dap | candidate_i | candidate_n | safe_i_at_target | safe_n_at_target | final_grnwt | total_irrigation | total_n | simple_profit | delta_vs_noop_simple_profit | delta_vs_background_simple_profit | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40.0000 | 80.0000 | 40.0000 | 80.0000 | 10908.5168 | 160.0000 | 250.0000 | 9498.5168 | -308.4607 | -0.0867 | 0 | 0 |
| 30 | 30.0000 | 50.0000 | 30.0000 | 50.0000 | 10908.6035 | 160.0000 | 250.0000 | 9498.6035 | -285.8350 | 0.0000 | 0 | 0 |
| 50 | 40.0000 | 80.0000 | 40.0000 | 80.0000 | 10906.8250 | 160.0000 | 250.0000 | 9496.8250 | -320.1428 | -1.7786 | 0 | 0 |
| 65 | 40.0000 | 80.0000 | 40.0000 | 80.0000 | 10891.7200 | 160.0000 | 250.0000 | 9481.7200 | -375.7312 | -16.8835 | 0 | 0 |
| 85 | 40.0000 | 80.0000 | 40.0000 | 50.0000 | 10900.6580 | 160.0000 | 250.0000 | 9490.6580 | -97.5537 | -7.9456 | 0 | 0 |
| 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10908.6035 | 160.0000 | 250.0000 | 9498.6035 | 0.0000 | 0.0000 | 0 | 0 |

## Full summary CSV

`benchmark_results/031_07_sy2014_free_timing_action_marginal_value_audit/evaluation/031_07_action_marginal_value_summary.csv`

## Figure

`benchmark_results/031_07_sy2014_free_timing_action_marginal_value_audit/figures/031_07_sy2014_action_marginal_profit_heatmap.png`

## Main finding

Under the fixed high-performing background schedule, the marginal-value map does not support early cap saturation.

- At DAP1, DAP30, DAP50, and DAP65, the simple-profit best branch is `I0/N0` at the target day. Adding irrigation or nitrogen at these individual target days either adds cost with negligible yield gain or reduces profit.
- At DAP85, the simple-profit best branch is `I20/N0`: water at this stage is useful, but extra nitrogen is not justified by final profit.
- At DAP100, nitrogen requests are clipped by the safety wrapper (`anfer_dap_range`), and irrigation at this point does not improve final profit under the common background.
- This supports the diagnosis that the PPO early-dump behavior in 031_06 is not a DSSAT-confirmed marginal-value optimum. It is a reward/learning-signal problem: the agent is not being guided toward the sparse timing structure revealed by the counterfactual map.

Concise target-DAP summary:

| Target DAP | Best simple-profit action | Final grain | Total I | Total N | Profit | Interpretation |
|---:|---|---:|---:|---:|---:|---|
| 1 | I0/N0 | 10936.98 | 130 | 200 | 9806.98 | Early operation is not needed under this background. |
| 30 | I0/N0 | 10914.44 | 130 | 200 | 9784.44 | Extra N can slightly raise yield but loses profit. |
| 50 | I0/N0 | 10946.97 | 130 | 200 | 9816.97 | Same pattern: yield response to extra N is too small for its cost. |
| 65 | I0/N0 | 10987.45 | 130 | 200 | 9857.45 | DAP65 extra N is especially not profit-justified. |
| 85 | I20/N0 | 10908.60 | 160 | 200 | 9748.60 | Late-season water is useful; N is not. |
| 100 | I0/N0 | 10908.60 | 160 | 250 | 9498.60 | No useful extra action; N is clipped after DAP90. |

## Interpretation boundary

This is not an RL result. It is a counterfactual marginal-value map for deciding how to redesign free-timing RL reward and diagnostics.
