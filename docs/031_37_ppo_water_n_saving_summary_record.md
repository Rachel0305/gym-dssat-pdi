# 031_37 PPO water/nitrogen saving summary record

## Scope

- No PPO/DQN training.
- No DSSAT rerun.
- Read 031_36 completed four-baseline comparison table and completed template-aware baseline table.
- Main saving baseline is `official_extension_expert`; four-baseline metric wins still use the 031_36 strict winner flags.

## Candidate selection rule

One display candidate per station-year was selected by: metric win count, advisor-any-win flag, profit, yield, lower irrigation, lower nitrogen, lower seed.

## Station summary

| station_code | years | selected_any_metric_win_years | selected_yield_win_years | selected_wp_et_win_years | selected_pfp_n_win_years | mean_irrigation_saving_vs_official_mm | median_irrigation_saving_vs_official_mm | min_irrigation_saving_vs_official_mm | max_irrigation_saving_vs_official_mm | mean_nitrogen_saving_vs_official_kg_ha | median_nitrogen_saving_vs_official_kg_ha | min_nitrogen_saving_vs_official_kg_ha | max_nitrogen_saving_vs_official_kg_ha | all_seed_rows | all_seed_any_metric_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 19 | 3 | 1 | 3 | 1 | 66.095 | 67.0 | 42.8 | 91.0 | 6.158 | 8.0 | -26.0 | 8.0 | 57 | 6 |
| HLA | 20 | 9 | 8 | 1 | 0 | 196.555 | 193.0 | 109.0 | 230.1 | -25.35 | -52.0 | -79.0 | 60.0 | 60 | 10 |
| LCA | 19 | 19 | 10 | 5 | 19 | 120.674 | 121.0 | 115.0 | 139.0 | 56.368 | 48.0 | 8.0 | 88.0 | 57 | 55 |
| YCA | 20 | 20 | 7 | 0 | 20 | 143.19 | 163.0 | 49.0 | 193.0 | 35.95 | 48.0 | 7.0 | 48.0 | 60 | 54 |

## Figure outputs

| figure |
| --- |
| benchmark_results/031_37_ppo_water_n_saving_summary/figures/031_37_fqa_ppo_water_n_saving_vs_official.png |
| benchmark_results/031_37_ppo_water_n_saving_summary/figures/031_37_hla_ppo_water_n_saving_vs_official.png |
| benchmark_results/031_37_ppo_water_n_saving_summary/figures/031_37_lca_ppo_water_n_saving_vs_official.png |
| benchmark_results/031_37_ppo_water_n_saving_summary/figures/031_37_yca_ppo_water_n_saving_vs_official.png |
| benchmark_results/031_37_ppo_water_n_saving_summary/figures/031_37_overall_ppo_water_n_saving_vs_official.png |

## Interpretation boundary

These plots summarize input saving relative to official expert. They do not by themselves prove the causal necessity of individual PPO actions.
