# 049_00 HLA originIC expanded-action MaskablePPO record

## Design

- Reference method: `046_10_sya_originIC_expanded_action_maskableppo`.
- Controlled change: station/input years only; PPO observation, reward, safety, seed, and action grid are inherited.
- Station/site: `HLA` / `HLA`.
- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`.
- Train years: `[2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]`.
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

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs                              |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|-----------------------------:|-------------------------:|------------------------------:|:--------------------------------------------------|
|              1000 |   2014 | True           |         146 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        5 |                             6 | I0/N80; I15/N40; I30/N0; I30/N40; I30/N80; I45/N0 |
|              1000 |   2015 | True           |         162 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        5 |                             6 | I0/N80; I15/N40; I30/N0; I30/N40; I30/N80; I45/N0 |
|              1000 |   2016 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             6 | I0/N40; I0/N80; I30/N0; I30/N40; I45/N0; I45/N80  |
|              1000 |   2017 | True           |         147 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             6 | I0/N40; I0/N80; I30/N0; I30/N40; I45/N0; I45/N80  |
|              1000 |   2018 | True           |         134 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             6 | I0/N40; I0/N80; I30/N0; I30/N40; I45/N0; I45/N80  |
|              1000 |   2019 | True           |         150 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             6 | I0/N40; I0/N80; I30/N0; I30/N40; I45/N0; I45/N80  |
|              1000 |   2020 | True           |         142 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        5 |                             4 | I0/N80; I30/N0; I30/N40; I45/N0                   |
|              1000 |   2021 | True           |         137 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        5 |                             6 | I0/N80; I15/N40; I30/N0; I30/N40; I30/N80; I45/N0 |
|              1000 |   2022 | True           |         141 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        5 |                             5 | I0/N80; I15/N120; I30/N0; I30/N40; I45/N0         |
|              1000 |   2023 | True           |         135 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        4 |                             6 | I0/N40; I0/N80; I30/N0; I30/N40; I45/N0; I45/N80  |
|              2000 |   2014 | True           |         146 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2015 | True           |         162 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2016 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2017 | True           |         147 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2018 | True           |         134 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2019 | True           |         150 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2020 | True           |         142 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2021 | True           |         137 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2022 | True           |         141 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |
|              2000 |   2023 | True           |         135 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        2 |                             4 | I0/N80; I30/N0; I45/N0; I45/N80                   |

## Normalized Outputs

- `evaluation/049_00_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/049_00_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/049_00_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `049_00_engine_result.json` <- `042_10_result.json`
