# 031_42 Sample free-timing PPO five-scenario daily process audit record

## Scope

This is a plotting and evidence-audit task only. No new training, no DSSAT rerun, no checkpoint reselection, and no reward or hyperparameter change were performed.

## Important limitation

The available 031_34/031_35/031_36 daily CSV sources do not include a complete soil-water storage column. The previous soil-water panel was therefore replaced by cumulative irrigation. SWTD was not inferred or fabricated.

## Fixed sample list

| station_code   |   year | reason                                                             |
|:---------------|-------:|:-------------------------------------------------------------------|
| HLA            |   2008 | HLA candidate with complete five-scenario daily sources            |
| HLA            |   2023 | HLA later-year candidate with complete five-scenario daily sources |
| LCA            |   2012 | LC candidate with moderate water and N                             |
| LCA            |   2023 | LC candidate with lower N                                          |
| YCA            |   2015 | YC candidate with relatively low irrigation and complete sources   |
| YCA            |   2023 | YC later-year candidate with complete sources                      |

## Coverage

- Successful plotted samples: 6/6

| station_code   |   year | status   | selected_row_found   | ppo_daily_exists   |   baseline_03135_rows |   dssat_auto_03136_rows | ppo_daily_path                                                                                                               |
|:---------------|-------:|:---------|:---------------------|:-------------------|----------------------:|------------------------:|:-----------------------------------------------------------------------------------------------------------------------------|
| HLA            |   2008 | ok       | True                 | True               |                   447 |                     149 | benchmark_results\031_34_four_site_all_year_frozen_maskableppo_transfer\daily_outputs\HLA\HLA_2008_seed0_ckpt75000_daily.csv |
| HLA            |   2023 | ok       | True                 | True               |                   414 |                     138 | benchmark_results\031_34_four_site_all_year_frozen_maskableppo_transfer\daily_outputs\HLA\HLA_2023_seed0_ckpt75000_daily.csv |
| LCA            |   2012 | ok       | True                 | True               |                   351 |                     117 | benchmark_results\031_34_four_site_all_year_frozen_maskableppo_transfer\daily_outputs\LCA\LCA_2012_seed1_ckpt75000_daily.csv |
| LCA            |   2023 | ok       | True                 | True               |                   264 |                      88 | benchmark_results\031_34_four_site_all_year_frozen_maskableppo_transfer\daily_outputs\LCA\LCA_2023_seed0_ckpt30000_daily.csv |
| YCA            |   2015 | ok       | True                 | True               |                   324 |                     108 | benchmark_results\031_34_four_site_all_year_frozen_maskableppo_transfer\daily_outputs\YCA\YCA_2015_seed1_ckpt10000_daily.csv |
| YCA            |   2023 | ok       | True                 | True               |                   291 |                      97 | benchmark_results\031_34_four_site_all_year_frozen_maskableppo_transfer\daily_outputs\YCA\YCA_2023_seed1_ckpt10000_daily.csv |

## Summary of sampled PPO candidates

| station_code   |   year |   final_grain_kg_ha |   total_irrigation_mm |   total_nitrogen_kg_ha |   irrigation_event_count |   six_mm_irrigation_count |   nitrogen_event_count |   max_water_stress_index_wspd |   max_nitrogen_stress_index_nstd |   stress_days_wspd_gt_0p05 |   stress_days_nstd_gt_0p05 |   first_irrigation_dap |   first_n_dap |
|:---------------|-------:|--------------------:|----------------------:|-----------------------:|-------------------------:|--------------------------:|-----------------------:|------------------------------:|---------------------------------:|---------------------------:|---------------------------:|-----------------------:|--------------:|
| HLA            |   2008 |             7110.22 |                    36 |                    240 |                        6 |                         6 |                      6 |                      0        |                        0.0515939 |                          0 |                          5 |                      1 |             1 |
| HLA            |   2023 |             6078.45 |                    36 |                    240 |                        6 |                         6 |                      6 |                      0        |                        0.0142938 |                          0 |                          0 |                      1 |             1 |
| LCA            |   2012 |             9018.83 |                    96 |                    200 |                       16 |                        16 |                      3 |                      0        |                        0.0121911 |                          0 |                          0 |                      1 |             2 |
| LCA            |   2023 |             7928.02 |                    72 |                    160 |                       12 |                        12 |                      1 |                      0.118229 |                        0.147035  |                          1 |                          6 |                      1 |             2 |
| YCA            |   2015 |             9088.61 |                    48 |                    200 |                        3 |                         0 |                      2 |                      0        |                        0.159856  |                          0 |                          9 |                      2 |             1 |
| YCA            |   2023 |             9574.11 |                    36 |                    200 |                        2 |                         0 |                      2 |                      0        |                        0.286668  |                          0 |                         14 |                      2 |             1 |

## Interpretation

- These figures are intended for visual inspection of whether the frozen PPO candidate actions look agronomically plausible against weather and stress trajectories.
- They are not causal proof that each action was necessary.
- Questionable events should be followed by same-prefix counterfactual DSSAT audits, e.g. delete or downgrade one action and compare final yield/resource metrics.