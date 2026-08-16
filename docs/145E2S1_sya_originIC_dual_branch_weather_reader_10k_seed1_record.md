# 145E2S1 SYA engineered forecast MaskablePPO record

## Design

- Reference run: `144E2_sya_originIC_dual_branch_weather_reader_10k_seed0`.
- Controlled change: append normalized engineered short-term weather-forecast summaries to the PPO observation.
- Unchanged: base DSSAT observation, reward/safety logic, action grid, train/validation years, seed, and input profile.
- Forecast mode: `rolling_7d_actionable_prefix_encoding`.
- Forecast source: `historical_WTH_as_perfect_rolling_7d_forecast`.
- Forecast features: `['rain_next1_norm', 'rain_next3_norm', 'rain_next7_norm', 'rain_max_next7_norm', 'rain_days_next7_norm', 'first_rain_lead_norm', 'tmax_mean_next7_norm', 'tmax_max_next7_norm', 'tmin_mean_next7_norm', 'srad_mean_next7_norm', 'valid_days_next7_norm', 'dry_spell_next7_flag']`.
- Observation smoke audit: `benchmark_results/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1/audits/145E2S1_forecast_observation_smoke.csv`.
- Forecast calendar audit: `benchmark_results/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1/audits/145E2S1_forecast_calendar_audit.csv`.
- Output root: `benchmark_results/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1`.

## independent seed1 10K gate

- `final_checkpoint`: `10000`
- `expected_validation_rows`: `10`
- `observed_validation_rows`: `10`
- `all_daily_files_exist`: `True`
- `all_actions_on_declared_grid`: `True`
- `positive_actions_present`: `True`
- `not_all_positive_actions_at_dap1`: `True`
- `forecast_columns_present`: `True`
- `forecast_columns_vary_within_seasons`: `True`
- `next_step_allowed`: `True`

## Formal smoke verification

- `seed0_audit_exists`: `True`
- `seed0_audit_completed`: `True`
- `requested_seed_matches`: `True`
- `only_seed_differs`: `True`
- `longer_runs_forbidden`: `True`
- `next_step_allowed`: `True`

## Year-level action and forecast-column audit

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   forecast_required_column_count |   forecast_present_column_count |   forecast_nonconstant_column_count | forecast_missing_columns   |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|---------------------------------:|--------------------------------:|------------------------------------:|:---------------------------|
|              2000 |   2014 | True           |         144 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2015 | True           |         141 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2016 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2017 | True           |         130 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2018 | True           |         129 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2019 | True           |         140 |                          0 |                        0 |                     15 |                                14 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2020 | True           |         138 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2021 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2022 | True           |         142 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|              2000 |   2023 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2014 | True           |         144 |                          0 |                        0 |                     15 |                                14 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2015 | True           |         141 |                          0 |                        0 |                     18 |                                17 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2016 | True           |         141 |                          0 |                        0 |                     18 |                                17 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2017 | True           |         130 |                          0 |                        0 |                     18 |                                17 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2018 | True           |         129 |                          0 |                        0 |                     18 |                                17 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2019 | True           |         140 |                          0 |                        0 |                     17 |                                16 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2020 | True           |         138 |                          0 |                        0 |                     18 |                                17 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2021 | True           |         138 |                          0 |                        0 |                     18 |                                17 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2022 | True           |         142 |                          0 |                        0 |                     19 |                                18 |                               24 |                              24 |                                  22 |                            |
|              5000 |   2023 | True           |         138 |                          0 |                        0 |                     16 |                                15 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2014 | True           |         144 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2015 | True           |         141 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2016 | True           |         141 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2017 | True           |         130 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2018 | True           |         129 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2019 | True           |         140 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2020 | True           |         138 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2021 | True           |         138 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2022 | True           |         142 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |
|             10000 |   2023 | True           |         138 |                          0 |                        0 |                     14 |                                13 |                               24 |                              24 |                                  22 |                            |

## Normalized outputs

- `evaluation/145E2S1_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/145E2S1_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/145E2S1_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `145E2S1_engine_result.json` <- `042_10_result.json`
