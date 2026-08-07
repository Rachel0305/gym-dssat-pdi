# 053_00 LCA/LC lowIC expanded-action MaskablePPO record

## Design

- Reference method: `046_10_sya_originIC_expanded_action_maskableppo`.
- Controlled change: station/input years only; PPO observation, reward, safety, seed, and action grid are inherited.
- Station/site: `LCA` / `LC`.
- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`.
- Train years: `[2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]`.
- Validation years: `[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]`.
- Training: `100000` steps, checkpoints `[25000, 50000, 75000, 100000]`.

## 100K Formal Action Audit

- `final_checkpoint`: `100000`
- `expected_validation_rows`: `10`
- `observed_validation_rows`: `10`
- `all_daily_files_exist`: `True`
- `all_actions_on_declared_grid`: `True`
- `positive_actions_present`: `True`
- `positive_actions_transmitted`: `True`
- `not_all_positive_actions_at_dap1`: `True`
- `multiple_nonzero_action_pairs`: `True`
- `new_action_levels_used`: `True`
- `next_step_allowed`: `True`

## Formal Smoke Verification

- `smoke_result_exists`: `True`
- `smoke_manifest_exists`: `True`
- `smoke_status_completed`: `True`
- `smoke_gate_passed`: `True`
- `smoke_timesteps_match`: `True`
- `smoke_checkpoints_match`: `True`
- `smoke_input_profile_match`: `True`
- `smoke_station_match`: `True`
- `smoke_actions_match`: `True`
- `smoke_observation_contract_match`: `True`
- `smoke_output_root`: `benchmark_results/053_00_lca_lowIC_expanded_action_maskableppo_smoke2k`
- `next_step_allowed`: `True`

## Year-Level Action Audit

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs                     |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|-----------------------------:|-------------------------:|------------------------------:|:-----------------------------------------|
|             25000 |   2014 | True           |         102 |                          0 |                        0 |                      5 |                                 4 |                            0 |                        4 |                             4 | I0/N40; I0/N80; I15/N0; I45/N120         |
|             25000 |   2015 | True           |          99 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        6 |                             4 | I0/N40; I0/N80; I15/N0; I45/N120         |
|             25000 |   2016 | True           |          93 |                          0 |                        0 |                      6 |                                 5 |                            0 |                        5 |                             4 | I0/N40; I0/N80; I15/N0; I45/N120         |
|             25000 |   2017 | True           |          94 |                          0 |                        0 |                      5 |                                 4 |                            0 |                        4 |                             4 | I0/N40; I0/N80; I15/N0; I45/N120         |
|             25000 |   2018 | True           |          87 |                          0 |                        0 |                      5 |                                 4 |                            0 |                        4 |                             4 | I0/N40; I0/N80; I15/N0; I45/N120         |
|             25000 |   2019 | True           |          92 |                          0 |                        0 |                      5 |                                 4 |                            0 |                        4 |                             4 | I0/N40; I0/N80; I15/N0; I45/N120         |
|             25000 |   2020 | True           |          92 |                          0 |                        0 |                      5 |                                 4 |                            0 |                        4 |                             4 | I0/N40; I0/N80; I15/N0; I45/N120         |
|             25000 |   2021 | True           |          96 |                          0 |                        0 |                      6 |                                 5 |                            0 |                        5 |                             5 | I0/N40; I0/N80; I15/N0; I30/N0; I45/N120 |
|             25000 |   2022 | True           |          95 |                          0 |                        0 |                      6 |                                 5 |                            0 |                        5 |                             5 | I0/N40; I0/N80; I15/N0; I30/N0; I45/N120 |
|             25000 |   2023 | True           |          88 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        6 |                             5 | I0/N40; I0/N80; I15/N0; I30/N0; I45/N120 |
|             50000 |   2014 | True           |         102 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        5 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2015 | True           |          99 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        5 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2016 | True           |          93 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2017 | True           |          94 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2018 | True           |          87 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2019 | True           |          92 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2020 | True           |          92 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2021 | True           |          96 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        5 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2022 | True           |          95 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             50000 |   2023 | True           |          88 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40  |
|             75000 |   2014 | True           |         102 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2015 | True           |          99 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2016 | True           |          93 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2017 | True           |          94 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2018 | True           |          87 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2019 | True           |          92 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2020 | True           |          92 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2021 | True           |          96 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2022 | True           |          95 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|             75000 |   2023 | True           |          88 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        6 |                             4 | I0/N120; I15/N0; I45/N0; I45/N120        |
|            100000 |   2014 | True           |         102 |                          0 |                        0 |                     13 |                                12 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2015 | True           |          99 |                          0 |                        0 |                     13 |                                12 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2016 | True           |          93 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2017 | True           |          94 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2018 | True           |          87 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2019 | True           |          92 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2020 | True           |          92 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2021 | True           |          96 |                          0 |                        0 |                     13 |                                12 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2022 | True           |          95 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |
|            100000 |   2023 | True           |          88 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             4 | I0/N40; I15/N0; I30/N0; I45/N0           |

## Normalized Outputs

- `evaluation/053_00_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/053_00_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/053_00_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `053_00_engine_result.json` <- `042_10_result.json`
