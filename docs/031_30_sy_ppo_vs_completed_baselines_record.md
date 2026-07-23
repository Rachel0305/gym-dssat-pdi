# 031_30 SY PPO versus completed baselines record

## Scope

- No training and no DSSAT rerun.
- 031_27 PPO candidates are compared against the 031_29 generated baseline envelope.
- `recorded_farmer_template_*` scenarios are treated as counterfactual historical-management transfers, not real recorded-farmer observations for the target year.

## Row-count checks

- Candidate rows: 57 / expected 57.
- Baseline rows: 114 / expected 114.

## Best-candidate year-level outcome

| years_with_any_seed_any_metric_winner | years_with_best_seed_yield_winner | years_with_best_seed_wp_et_winner | years_with_best_seed_pfp_n_winner | years_with_best_seed_all_three_winner | total_years |
| --- | --- | --- | --- | --- | --- |
| 17 | 5 | 0 | 17 | 0 | 19 |

## Seed-level summary

| seed | year_count | any_metric_winner_years | yield_winner_years | wp_et_winner_years | pfp_n_winner_years | all_three_winner_years | mean_yield_delta_vs_best_baseline | mean_wp_et_delta_vs_best_baseline | mean_pfp_n_delta_vs_best_baseline | mean_water_saving_vs_official_expert | mean_n_saving_vs_official_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 19 | 16 | 5 | 0 | 16 | 0 | -110.2276 | -0.1437 | 1.7474 | 158.0 | 60.0 |
| 1 | 19 | 17 | 4 | 0 | 17 | 0 | -102.231 | -0.1395 | 1.7737 | 175.0526 | 60.0 |
| 2 | 19 | 17 | 5 | 0 | 17 | 0 | -96.229 | -0.1411 | 1.8053 | 158.0 | 60.0 |

## Best candidate by year

| year | seed | checkpoint_step | final_grain_kg_ha | wp_et_kg_m3 | pfp_n_kg_kg | total_irrigation | total_n | n_winning_metrics | yield_winner | wp_et_winner | pfp_n_winner | yield_delta_vs_best_baseline | wp_et_delta_vs_best_baseline | pfp_n_delta_vs_best_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | 2 | 50000 | 9429.5422 | 1.97 | 39.3 | 108.0 | 240.0 | 1 | False | False | True | -237.4578 | -0.13 | 0.2 |
| 2006 | 0 | 20000 | 10340.9033 | 2.17 | 43.1 | 108.0 | 240.0 | 1 | False | False | True | -47.0967 | -0.17 | 1.0 |
| 2007 | 2 | 50000 | 10887.2522 | 2.43 | 45.4 | 108.0 | 240.0 | 1 | False | False | True | -394.7478 | -0.23 | 1.6 |
| 2008 | 1 | 100000 | 10806.3257 | 2.19 | 45.0 | 90.0 | 240.0 | 0 | False | False | False | -338.6743 | -0.19 | -0.1 |
| 2009 | 2 | 50000 | 12657.1399 | 2.51 | 52.7 | 108.0 | 240.0 | 2 | True | False | True | 302.1399 | -0.1 | 4.5 |
| 2010 | 1 | 100000 | 8432.55 | 1.85 | 35.1 | 96.0 | 240.0 | 1 | False | False | True | -71.45 | -0.14 | 0.7 |
| 2011 | 1 | 100000 | 9497.6276 | 2.01 | 39.6 | 96.0 | 240.0 | 1 | False | False | True | -55.3724 | -0.13 | 0.9 |
| 2012 | 0 | 20000 | 9982.5977 | 2.14 | 41.6 | 108.0 | 240.0 | 1 | False | False | True | -184.4023 | -0.15 | 0.4 |
| 2013 | 1 | 100000 | 10698.3508 | 2.29 | 44.6 | 90.0 | 240.0 | 1 | False | False | True | -7.6492 | -0.18 | 1.3 |
| 2014 | 1 | 100000 | 10943.8074 | 2.14 | 45.6 | 96.0 | 240.0 | 2 | True | False | True | 143.8074 | -0.03 | 7.2 |
| 2015 | 1 | 100000 | 10977.2363 | 2.26 | 45.7 | 90.0 | 240.0 | 1 | False | False | True | -199.7637 | -0.19 | 0.4 |
| 2016 | 1 | 100000 | 7953.2983 | 1.64 | 33.1 | 96.0 | 240.0 | 1 | False | False | True | -80.7017 | -0.12 | 0.6 |
| 2017 | 0 | 20000 | 10913.1848 | 2.46 | 45.5 | 108.0 | 240.0 | 2 | True | False | True | 58.1848 | -0.03 | 9.2 |
| 2018 | 1 | 100000 | 8282.937 | 1.78 | 34.5 | 84.0 | 240.0 | 1 | False | False | True | -99.063 | -0.14 | 0.6 |
| 2019 | 1 | 100000 | 10423.9795 | 2.12 | 43.4 | 90.0 | 240.0 | 2 | True | False | True | 194.9795 | -0.07 | 2.0 |
| 2020 | 1 | 100000 | 9971.3135 | 2.08 | 41.5 | 90.0 | 240.0 | 2 | True | False | True | 146.3135 | -0.06 | 3.3 |
| 2021 | 0 | 20000 | 10319.1809 | 2.18 | 43.0 | 108.0 | 240.0 | 1 | False | False | True | -93.8191 | -0.12 | 0.8 |
| 2022 | 1 | 100000 | 10383.1104 | 2.21 | 43.3 | 90.0 | 240.0 | 0 | False | False | False | -580.8896 | -0.26 | -1.1 |
| 2023 | 0 | 20000 | 10706.1633 | 2.16 | 44.6 | 108.0 | 240.0 | 1 | False | False | True | -170.8367 | -0.19 | 1.2 |