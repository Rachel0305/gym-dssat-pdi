# 031_41 LC2010 operation-event-cost MaskablePPO smoke record

## Scope

- LC2010 only, seed1 only.
- Same free-timing MaskablePPO framework as 031_33.
- Only reward accounting changed: fixed operation costs were added.
- Action grid, seasonal caps, min interval, DAP ranges, PPO hyperparameters, weather, and DSSAT inputs were not changed.

## Operation-event-cost reward

```text
unscaled_reward = literature_yield_component - literature_resource_cost - operation_event_cost
operation_event_cost = 10 * 1[irrigation > 0] + 20 * 1[nitrogen > 0]
reward = unscaled_reward * 0.001
```

## Pre-registered config

```json
{
  "base_config": "experiments/ppo_observed_years/config_031_20_free_timing_discrete_maskableppo_scaled_reward_five_station_smoke.yaml",
  "output_root": "benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke",
  "target_site_year": {
    "station_code": "LCA",
    "year": 2010
  },
  "seed": 1,
  "max_train_steps": 50000,
  "checkpoint_steps": [
    10000,
    30000,
    50000
  ],
  "operation_event_cost": {
    "irrigation_event_cost_unscaled": 10.0,
    "fertilization_event_cost_unscaled": 20.0
  },
  "selection_metric": "operation_adjusted_reward_sum",
  "tie_breakers": [
    "final_grnwt_desc",
    "irrigation_event_count_asc",
    "total_irrigation_asc",
    "total_n_asc",
    "checkpoint_step_asc"
  ],
  "runtime_notes": {
    "no_parameter_scan": true,
    "single_site_year_smoke_only": true,
    "action_grid_unchanged": true,
    "action_safety_unchanged": true,
    "no_expert_dap_windows": true
  }
}
```

## Training summary

station_code  year  seed  checkpoint_step  run_status                                                                                                                                model_path                                                     model_sha256
         LCA  2010     1            10000 ok_existing benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/models/LCA/LCA_2010_operation_cost_maskableppo_seed1_ckpt10000.zip c9649eed26c93566b80509af5439b1a4be5577da771cd8faa699bcb953f5b51a
         LCA  2010     1            30000 ok_existing benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/models/LCA/LCA_2010_operation_cost_maskableppo_seed1_ckpt30000.zip 05e803a507bac31d6b2407a940ed10a551a537aed2c69ec92bcbc186e4f5a3af
         LCA  2010     1            50000 ok_existing benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/models/LCA/LCA_2010_operation_cost_maskableppo_seed1_ckpt50000.zip b36d479645a62d4f9ee97d3f6d1530103a9759282384e404e8251e22ad65089c

## Checkpoint evaluation summary

station_code  year  seed  checkpoint_step           split run_status  episode_length  final_grnwt  final_biomass  total_irrigation  total_n  WP_ET_kg_m3  yield_per_irrigation_mm_proxy     PFP_N  profit_simple  operation_adjusted_reward_sum  operation_event_cost_total_unscaled  reward_closure_max_abs_error  irrigation_event_count  n_event_count  six_mm_irrigation_count  small_irrigation_le_6mm_count  first_irrigation_dap  first_n_dap  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                         action_sequence                                                                                                                                model_path                                                                                                                             daily_csv_path
         LCA  2010     1            10000 train_year_eval         ok              95  8327.671509   16178.116455              24.0    240.0          NaN                     346.986313 34.698631    7103.671509                       0.820172                                 90.0                           0.0                       3              3                        2                              2                     1            1                          1                    0 DAP1 I12/N80; DAP8 I6/N80; DAP15 I6/N80 benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/models/LCA/LCA_2010_operation_cost_maskableppo_seed1_ckpt10000.zip benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/daily_outputs/LCA/LCA_2010_seed1_ckpt10000_operation_cost_daily.csv
         LCA  2010     1            30000 train_year_eval         ok              95  7908.199463   15849.732666              18.0    120.0          NaN                     439.344415 65.901662    7290.199463                       0.950096                                 90.0                           0.0                       3              3                        3                              3                     2            2                          1                   17  DAP2 I6/N40; DAP8 I6/N40; DAP15 I6/N40 benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/models/LCA/LCA_2010_operation_cost_maskableppo_seed1_ckpt30000.zip benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/daily_outputs/LCA/LCA_2010_seed1_ckpt30000_operation_cost_daily.csv
         LCA  2010     1            50000 train_year_eval         ok              95  8323.299561   16214.996338               6.0    160.0          NaN                    1387.216593 52.020622    7517.299561                       1.025681                                 30.0                           0.0                       1              1                        1                              1                     2            2                          0                    0                            DAP2 I6/N160 benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/models/LCA/LCA_2010_operation_cost_maskableppo_seed1_ckpt50000.zip benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/daily_outputs/LCA/LCA_2010_seed1_ckpt50000_operation_cost_daily.csv

## Selected checkpoint

station_code year seed checkpoint_step           split run_status episode_length  final_grnwt final_biomass total_irrigation total_n WP_ET_kg_m3 yield_per_irrigation_mm_proxy      PFP_N profit_simple operation_adjusted_reward_sum operation_event_cost_total_unscaled reward_closure_max_abs_error irrigation_event_count n_event_count six_mm_irrigation_count small_irrigation_le_6mm_count first_irrigation_dap first_n_dap swfac_stress_days_gt_0p05 nstres_days_gt_0p05 action_sequence                                                                                                                                model_path                                                                                                                             daily_csv_path
         LCA 2010    1           50000 train_year_eval         ok             95  8323.299561  16214.996338              6.0   160.0         NaN                   1387.216593  52.020622   7517.299561                      1.025681                                30.0                          0.0                      1             1                       1                             1                    2           2                         0                   0    DAP2 I6/N160 benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/models/LCA/LCA_2010_operation_cost_maskableppo_seed1_ckpt50000.zip benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/daily_outputs/LCA/LCA_2010_seed1_ckpt50000_operation_cost_daily.csv

## 031_39 reference context

site   station  year   algorithm  seed  checkpoint                  scenario  final_grain_kg_ha  final_biomass_kg_ha  rain_total_mm  irrigation_event_total_mm  nitrogen_event_total_kg_ha  max_water_stress_wspd  max_nitrogen_stress_nstd  common_reward_total                                                                                                                                 snapshot_path  etcp_mm  wp_et_kg_m3  pfp_n_kg_kg  n_uptake_kg_ha  n_leaching_kg_ha                                        source
  LC Luancheng  2010 MaskablePPO   NaN         NaN                       NaN             8051.0              15541.0          110.2                        0.0                         0.0                  0.504                     0.019                  0.0                      benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/readiness/baseline_runs/null/pdi_tmp_snapshot_eval    256.9         3.13          NaN           203.0               0.0 031_39_current_frozen_candidate_and_baselines
  LC Luancheng  2010 MaskablePPO   NaN         NaN           recorded_farmer             8732.0              16324.0          110.2                      130.0                       250.0                  0.000                     0.019               -699.0           benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/readiness/baseline_runs/recorded_farmer/pdi_tmp_snapshot_eval    286.2         3.05         34.9           213.0               0.0 031_39_current_frozen_candidate_and_baselines
  LC Luancheng  2010 MaskablePPO   NaN         NaN                dssat_auto             8738.0              16373.0          110.2                      138.5                         0.0                  0.000                     0.019                548.5                benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/readiness/baseline_runs/dssat_auto/pdi_tmp_snapshot_eval    286.8         3.05          NaN           214.0               0.0 031_39_current_frozen_candidate_and_baselines
  LC Luancheng  2010 MaskablePPO   NaN         NaN official_extension_expert             8739.0              16376.0          110.2                      198.8                       247.0                  0.000                     0.019               -745.8 benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/readiness/baseline_runs/official_extension_expert/pdi_tmp_snapshot_eval    281.9         3.10         35.3           214.0               0.0 031_39_current_frozen_candidate_and_baselines
  LC Luancheng  2010 MaskablePPO   1.0     75000.0              rl_candidate             8349.0              16444.0          255.1                       78.0                       160.0                  0.000                     0.012               -580.0                                benchmark_results/031_34_four_site_all_year_frozen_maskableppo_transfer/runs/LCA/2010/seed1_ckpt75000/snapshot    287.2         2.91         52.2           206.0               0.0 031_39_current_frozen_candidate_and_baselines

## Interpretation

- Reference PPO candidate total irrigation/event count: NA mm / NA events.
- 031_41 selected total irrigation/event count: 6.0 mm / 1 events.
- 031_41 selected six-mm irrigation count: 1.
- Reward closure max absolute error across evaluated checkpoints: 0.0.
- `WP_ET_kg_m3` is intentionally left blank in this smoke summary because formal WP_ET requires ET/ETCP extraction from DSSAT outputs; `yield_per_irrigation_mm_proxy` is only a quick irrigation-efficiency proxy and must not be reported as WP_ET.

This smoke does not authorize all-site reruns. It only tests whether fixed operation cost is a plausible way to reduce management-unrealistic repeated small irrigation.

## Post-hoc read-only audit: existing selected PPO small-irrigation prevalence

After the LC2010 smoke, a read-only audit was added to answer whether repeated 6 mm irrigation is unique to LC2010. This audit did not retrain any model and only read the selected-candidate daily CSV paths listed in `benchmark_results/031_38_five_station_ppo_water_n_saving_summary/tables/031_38_selected_station_year_ppo_deltas_vs_official.csv`.

Output table:

```text
benchmark_results/031_41_lc2010_operation_event_cost_maskableppo_smoke/tables/031_41_existing_selected_ppo_small_irrigation_audit.csv
```

Station-level result across 97 selected station-years:

```text
station_code  years  mean_i_events  mean_six  years_with_6mm  max_six
FQA              19      13.263158  7.631579              19       12
HLA              20       6.400000  6.250000              20       11
LCA              19      13.526316 13.473684              19       16
SYA              19      15.526316 15.526316              19       18
YCA              20       4.950000  1.950000               6       14
```

Interpretation: LC2010 is not an isolated case. Under the current `[0, 6, 12, 18, 24]` irrigation action grid, frequent 6 mm irrigation is common in selected PPO candidates, especially SYA/LCA/FQA. This supports treating the issue as an action/reward design problem rather than a one-year visual oddity.

