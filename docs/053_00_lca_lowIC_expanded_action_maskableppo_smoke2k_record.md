# 053_00 LCA/LC lowIC expanded-action MaskablePPO record

## Design

- Reference method: `046_10_sya_originIC_expanded_action_maskableppo`.
- Controlled change: station/input years only; PPO observation, reward, safety, seed, and action grid are inherited.
- Station/site: `LCA` / `LC`.
- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`.
- Train years: `[2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]`.
- Validation years: `[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]`.
- Training: `2000` steps, checkpoints `[1000, 2000]`.

## 2K Smoke Action Audit

- `final_checkpoint`: `2000`
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

## Year-Level Action Audit

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs                      |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|-----------------------------:|-------------------------:|------------------------------:|:------------------------------------------|
|              1000 |   2014 | True           |         102 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2015 | True           |          99 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2016 | True           |          93 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2017 | True           |          94 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2018 | True           |          87 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2019 | True           |          92 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2020 | True           |          92 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N40   |
|              1000 |   2021 | True           |          96 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2022 | True           |          95 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              1000 |   2023 | True           |          88 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        4 |                             5 | I0/N40; I0/N80; I30/N0; I45/N0; I45/N120  |
|              2000 |   2014 | True           |         102 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        8 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2015 | True           |          99 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        8 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2016 | True           |          93 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        7 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2017 | True           |          94 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        7 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2018 | True           |          87 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        7 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2019 | True           |          92 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        7 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2020 | True           |          92 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        7 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2021 | True           |          96 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        8 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2022 | True           |          95 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        7 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |
|              2000 |   2023 | True           |          88 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        7 |                             5 | I0/N80; I15/N0; I15/N40; I30/N0; I45/N120 |

## Normalized Outputs

- `evaluation/053_00_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/053_00_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/053_00_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `053_00_engine_result.json` <- `042_10_result.json`
