# 031_39 representative free-timing PPO five-scenario daily record

## Scope

- No training.
- No new DSSAT run.
- Current 031-series free-timing MaskablePPO candidate for LCA2010.
- Plot style reuses the 027_05 five-scenario daily process design.

## Selected candidate

| station_code | site | year | seed | checkpoint_step | final_grain_kg_ha | irrigation_mm | nitrogen_kg_ha | winning_metrics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LCA | LC | 2010 | 1 | 75000 | 8348.992 | 78.0 | 160.0 | PFP_N |

## Daily evidence checks

| site | algorithm | scenario | check | passed | value |
| --- | --- | --- | --- | --- | --- |
| LC | MaskablePPO | null | snapshot_exists | True | benchmark_results\027_07_site_specific_stage_maskable_ppo_attempt2\LC\readiness\baseline_runs\null\pdi_tmp_snapshot_eval |
| LC | MaskablePPO | null | daily_irrigation_matches_summary_event_total | True | 0.0 |
| LC | MaskablePPO | null | daily_nitrogen_matches_summary_event_total | True | 0.0 |
| LC | MaskablePPO | null | rain_not_all_zero_unless_weather_is_zero | True | 110.2 |
| LC | MaskablePPO | null | temperature_source_anomaly_count_documented | True | 0 |
| LC | MaskablePPO | recorded_farmer | snapshot_exists | True | benchmark_results\027_07_site_specific_stage_maskable_ppo_attempt2\LC\readiness\baseline_runs\recorded_farmer\pdi_tmp_snapshot_eval |
| LC | MaskablePPO | recorded_farmer | daily_irrigation_matches_summary_event_total | True | 130.0 |
| LC | MaskablePPO | recorded_farmer | daily_nitrogen_matches_summary_event_total | True | 250.0 |
| LC | MaskablePPO | recorded_farmer | rain_not_all_zero_unless_weather_is_zero | True | 110.2 |
| LC | MaskablePPO | recorded_farmer | temperature_source_anomaly_count_documented | True | 0 |
| LC | MaskablePPO | dssat_auto | snapshot_exists | True | benchmark_results\027_07_site_specific_stage_maskable_ppo_attempt2\LC\readiness\baseline_runs\dssat_auto\pdi_tmp_snapshot_eval |
| LC | MaskablePPO | dssat_auto | daily_irrigation_matches_summary_event_total | True | 138.5 |
| LC | MaskablePPO | dssat_auto | daily_nitrogen_matches_summary_event_total | True | 0.0 |
| LC | MaskablePPO | dssat_auto | rain_not_all_zero_unless_weather_is_zero | True | 110.2 |
| LC | MaskablePPO | dssat_auto | temperature_source_anomaly_count_documented | True | 0 |
| LC | MaskablePPO | official_extension_expert | snapshot_exists | True | benchmark_results\027_07_site_specific_stage_maskable_ppo_attempt2\LC\readiness\baseline_runs\official_extension_expert\pdi_tmp_snapshot_eval |
| LC | MaskablePPO | official_extension_expert | daily_irrigation_matches_summary_event_total | True | 198.8 |
| LC | MaskablePPO | official_extension_expert | daily_nitrogen_matches_summary_event_total | True | 247.0 |
| LC | MaskablePPO | official_extension_expert | rain_not_all_zero_unless_weather_is_zero | True | 110.2 |
| LC | MaskablePPO | official_extension_expert | temperature_source_anomaly_count_documented | True | 0 |
| LC | MaskablePPO | rl_candidate | snapshot_exists | True | benchmark_results\031_34_four_site_all_year_frozen_maskableppo_transfer\runs\LCA\2010\seed1_ckpt75000\snapshot |
| LC | MaskablePPO | rl_candidate | daily_irrigation_matches_summary_event_total | True | 78.0 |
| LC | MaskablePPO | rl_candidate | daily_nitrogen_matches_summary_event_total | True | 160.0 |
| LC | MaskablePPO | rl_candidate | rain_not_all_zero_unless_weather_is_zero | True | 255.09999999999997 |
| LC | MaskablePPO | rl_candidate | temperature_source_anomaly_count_documented | True | 0 |

## Outputs

| path |
| --- |
| benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/figures/027_05_lc2010_maskableppo_five_scenario_daily.png |
| benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/figures/027_05_lc2010_maskableppo_five_scenario_daily.svg |
| benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/figures/027_05_lc2010_ppo_five_scenario_endpoints.png |
| benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/figures/027_05_lc2010_ppo_five_scenario_endpoints.svg |
| benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/031_39_lca2010_current_free_timing_ppo_five_scenario_daily.csv |
| benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/031_39_lca2010_current_free_timing_ppo_five_scenario_summary.csv |
| benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/031_39_lca2010_daily_evidence_checks.csv |
