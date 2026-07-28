# 032_06 LC2010 guardrail-selected five-scenario daily record

## Scope

- No training.
- No new DSSAT runs.
- Four baseline scenarios reuse 031_39 daily evidence.
- PPO candidates use 032_05 guardrail-selected checkpoints from 032_04 daily outputs.
- The old cumulative reward panel is not reused; final panel shows cumulative irrigation and nitrogen.

## Selected candidates

| seed | checkpoint_step | final_grnwt | total_irrigation | total_n | max_nstres | action_sequence | daily_csv_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 50000 | 8337.897 | 45.0 | 160.0 | 0.012 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/daily_outputs/LCA/LCA_2010_seed0_ckpt50000_daily.csv |
| 1 | 10000 | 8317.422 | 135.0 | 240.0 | 0.012 | DAP1 I45/N120; DAP8 I45/N80; DAP16 I45/N0; DAP19 I0/N40 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/daily_outputs/LCA/LCA_2010_seed1_ckpt10000_daily.csv |
| 2 | 25000 | 8366.881 | 150.0 | 160.0 | 0.12 | DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP22 I45/N40; DAP29 I15/N0; DAP36 I15/N0; DAP63 I0/N80 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/daily_outputs/LCA/LCA_2010_seed2_ckpt25000_daily.csv |

## Five-scenario endpoint summaries

| seed | checkpoint_step | scenario | final_grain_kg_ha | irrigation_event_total_mm | nitrogen_event_total_kg_ha | PFP_N | max_water_stress_wspd | max_nitrogen_stress_nstd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 50000 | dssat_auto | 8738.0 | 138.5 | 0.0 |  | 0.0 | 0.019 |
| 0 | 50000 | null | 8051.0 | 0.0 | 0.0 |  | 0.504 | 0.019 |
| 0 | 50000 | official_extension_expert | 8739.0 | 198.8 | 247.0 | 35.381 | 0.0 | 0.019 |
| 0 | 50000 | recorded_farmer | 8732.0 | 130.0 | 250.0 | 34.928 | 0.0 | 0.019 |
| 0 | 50000 | rl_candidate | 8337.897 | 45.0 | 160.0 | 52.112 | 0.076 | 0.012 |
| 1 | 10000 | dssat_auto | 8738.0 | 138.5 | 0.0 |  | 0.0 | 0.019 |
| 1 | 10000 | null | 8051.0 | 0.0 | 0.0 |  | 0.504 | 0.019 |
| 1 | 10000 | official_extension_expert | 8739.0 | 198.8 | 247.0 | 35.381 | 0.0 | 0.019 |
| 1 | 10000 | recorded_farmer | 8732.0 | 130.0 | 250.0 | 34.928 | 0.0 | 0.019 |
| 1 | 10000 | rl_candidate | 8317.422 | 135.0 | 240.0 | 34.656 | 0.0 | 0.012 |
| 2 | 25000 | dssat_auto | 8738.0 | 138.5 | 0.0 |  | 0.0 | 0.019 |
| 2 | 25000 | null | 8051.0 | 0.0 | 0.0 |  | 0.504 | 0.019 |
| 2 | 25000 | official_extension_expert | 8739.0 | 198.8 | 247.0 | 35.381 | 0.0 | 0.019 |
| 2 | 25000 | recorded_farmer | 8732.0 | 130.0 | 250.0 | 34.928 | 0.0 | 0.019 |
| 2 | 25000 | rl_candidate | 8366.881 | 150.0 | 160.0 | 52.293 | 0.0 | 0.12 |

## Evidence checks

| seed | checkpoint | check | passed | value |
| --- | --- | --- | --- | --- |
| 0 | 50000 | base_daily_exists | True | benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/031_39_lca2010_current_free_timing_ppo_five_scenario_daily.csv |
| 0 | 50000 | ppo_daily_exists | True | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/daily_outputs/LCA/LCA_2010_seed0_ckpt50000_daily.csv |
| 0 | 50000 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 0 | 50000 | ppo_irrigation_matches_selection | True | 45.0 |
| 0 | 50000 | ppo_n_matches_selection | True | 160.0 |
| 1 | 10000 | base_daily_exists | True | benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/031_39_lca2010_current_free_timing_ppo_five_scenario_daily.csv |
| 1 | 10000 | ppo_daily_exists | True | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/daily_outputs/LCA/LCA_2010_seed1_ckpt10000_daily.csv |
| 1 | 10000 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 1 | 10000 | ppo_irrigation_matches_selection | True | 135.0 |
| 1 | 10000 | ppo_n_matches_selection | True | 240.0 |
| 2 | 25000 | base_daily_exists | True | benchmark_results/031_39_representative_free_timing_ppo_five_scenario_daily/031_39_lca2010_current_free_timing_ppo_five_scenario_daily.csv |
| 2 | 25000 | ppo_daily_exists | True | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/daily_outputs/LCA/LCA_2010_seed2_ckpt25000_daily.csv |
| 2 | 25000 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2 | 25000 | ppo_irrigation_matches_selection | True | 150.0 |
| 2 | 25000 | ppo_n_matches_selection | True | 160.0 |

## Outputs

| seed | checkpoint_step | daily_csv | summary_csv | checks_csv | figures |
| --- | --- | --- | --- | --- | --- |
| 0 | 50000 | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed0_five_scenario_daily.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed0_five_scenario_summary.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed0_checks.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/figures/032_06_lc2010_seed0_ckpt50000_five_scenario_daily.png;benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/figures/032_06_lc2010_seed0_ckpt50000_five_scenario_daily.svg |
| 1 | 10000 | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed1_five_scenario_daily.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed1_five_scenario_summary.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed1_checks.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/figures/032_06_lc2010_seed1_ckpt10000_five_scenario_daily.png;benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/figures/032_06_lc2010_seed1_ckpt10000_five_scenario_daily.svg |
| 2 | 25000 | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed2_five_scenario_daily.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed2_five_scenario_summary.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/tables/032_06_lc2010_seed2_checks.csv | benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/figures/032_06_lc2010_seed2_ckpt25000_five_scenario_daily.png;benchmark_results/032_06_lc2010_guardrail_selected_five_scenario_daily/figures/032_06_lc2010_seed2_ckpt25000_five_scenario_daily.svg |
