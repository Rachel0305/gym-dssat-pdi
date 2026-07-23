# 031_38 five-station PPO water/nitrogen saving summary record

## Scope

- No training.
- No DSSAT rerun.
- Adds SYA results from 031_30 to the 031_37 four-station summary.
- Water/nitrogen saving is relative to official expert.
- Metric-win markers are evaluated against the four-baseline envelope.

## Source tables

- `benchmark_results\031_37_ppo_water_n_saving_summary\tables\031_37_all_seed_ppo_deltas_vs_official.csv`
- `benchmark_results\031_37_ppo_water_n_saving_summary\tables\031_37_selected_station_year_ppo_deltas_vs_official.csv`
- `benchmark_results\031_30_sy_ppo_vs_completed_baselines\evaluation\031_30_candidate_vs_baseline_envelope.csv`

## Candidate display selection

One selected candidate per station-year is chosen by metric win count, any-win flag, profit, yield, lower irrigation, lower nitrogen, then lower seed id.

## Station summary

| station_code | years | selected_any_metric_win_years | selected_yield_win_years | selected_wp_et_win_years | selected_pfp_n_win_years | mean_irrigation_saving_vs_official_mm | sd_irrigation_saving_vs_official_mm | min_irrigation_saving_vs_official_mm | max_irrigation_saving_vs_official_mm | mean_nitrogen_saving_vs_official_kg_ha | sd_nitrogen_saving_vs_official_kg_ha | min_nitrogen_saving_vs_official_kg_ha | max_nitrogen_saving_vs_official_kg_ha | all_seed_rows | all_seed_any_metric_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 19 | 3 | 1 | 3 | 1 | 66.095 | 17.769 | 42.8 | 91.0 | 6.158 | 7.791 | -26.0 | 8.0 | 57 | 6 |
| HLA | 20 | 9 | 8 | 1 | 0 | 196.555 | 27.678 | 109.0 | 230.1 | -25.35 | 50.91 | -79.0 | 60.0 | 60 | 10 |
| LCA | 19 | 19 | 10 | 5 | 19 | 120.674 | 6.774 | 115.0 | 139.0 | 56.368 | 25.156 | 8.0 | 88.0 | 57 | 55 |
| SYA | 19 | 17 | 5 | 0 | 17 | 172.842 | 6.122 | 158.0 | 182.0 | 60.0 | 0.0 | 60.0 | 60.0 | 57 | 50 |
| YCA | 20 | 20 | 7 | 0 | 20 | 143.19 | 44.367 | 49.0 | 193.0 | 35.95 | 18.886 | 7.0 | 48.0 | 60 | 54 |

## Outputs

| figure |
| --- |
| benchmark_results\031_38_five_station_ppo_water_n_saving_summary\figures\031_38_fqa_ppo_water_n_saving_vs_official.png |
| benchmark_results\031_38_five_station_ppo_water_n_saving_summary\figures\031_38_hla_ppo_water_n_saving_vs_official.png |
| benchmark_results\031_38_five_station_ppo_water_n_saving_summary\figures\031_38_lca_ppo_water_n_saving_vs_official.png |
| benchmark_results\031_38_five_station_ppo_water_n_saving_summary\figures\031_38_sya_ppo_water_n_saving_vs_official.png |
| benchmark_results\031_38_five_station_ppo_water_n_saving_summary\figures\031_38_yca_ppo_water_n_saving_vs_official.png |
| benchmark_results\031_38_five_station_ppo_water_n_saving_summary\figures\031_38_overall_ppo_water_n_saving_vs_official.png |

## Interpretation boundary

The saving baseline is official expert because null and DSSAT auto may use zero water or zero nitrogen. Four-baseline maxima remain the correct benchmark for yield/WP_ET/PFP_N metric wins.
