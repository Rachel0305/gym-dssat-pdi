# 032_02 cumulative reward plot metric audit record

## Dataset and grain

- Source summary: `benchmark_results/031_42_sample_ppo_five_scenario_daily_process_audit/tables/031_42_sample_five_scenario_summary.csv`.
- Source daily: `benchmark_results/031_42_sample_ppo_five_scenario_daily_process_audit/tables/031_42_sample_five_scenario_daily.csv`.
- Intended daily grain: station-year-scenario-DAP.
- Daily rows: 3485. Summary rows: 30.

## Checks performed

- Recomputed endpoint common rewards from final grain, total irrigation, and total N.
- Compared source `final_cumulative_available_reward` against recomputed common rewards.
- Flagged mixed scale cases where PPO looks 0.001-scaled while baseline scenarios look unscaled or use a different reward definition.

## Key findings

- Mixed reward scale/definition detected in 6/6 sampled site-years.
- PPO source/common031 ratio range: 9.56941e-05 to 0.00012552.
- Non-PPO source/common031 ratio range: 1.62237 to 3.21275.
- In the 031_42 sample package, PPO cumulative reward is usually around 0.x because PPO daily reward is scaled, while baseline cumulative rewards are often thousands to tens of thousands from their own source reward columns.
- Therefore the existing cumulative reward panel is not a valid common-reward comparison across five scenarios.

## Impact

- Figures can misleadingly show PPO as having the lowest cumulative reward even when endpoint yield/water/N metrics are competitive.
- This is a plotting metric problem, not evidence that PPO optimized the worst reward.

## Recommendation

- Do not label the existing panel as `Cumulative common reward`.
- For advisor-facing daily process figures, either remove the reward panel or replace it with cumulative irrigation and cumulative nitrogen usage.
- If a reward comparison is needed, recompute a single endpoint common reward formula for all scenarios and show it as an endpoint bar/table, not by mixing source daily reward columns.

## Output tables

- Scenario audit: `benchmark_results/032_02_cumulative_reward_plot_metric_audit/tables/032_02_reward_metric_audit_by_scenario.csv`.
- Site-year audit: `benchmark_results/032_02_cumulative_reward_plot_metric_audit/tables/032_02_reward_metric_audit_by_site_year.csv`.
