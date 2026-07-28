# 032_09 LC historical/future split and DSSAT-auto gap audit record

## Scope

- No training.
- No DSSAT run.
- Station: LCA/LC.
- Candidate protocol: train on 2000-2010, test on 2011-2020, extension test on 2021-2023.

## Split and baseline coverage

| year | split | scenario_pool_available | weather_inventory_available | weather_clean_available | has_null | has_official_extension_expert | has_recorded_farmer | has_dssat_auto | five_scenario_plot_ready | needs_dssat_auto_completion | recommended_use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | train_2000_2010 | True | True | False | True | True | True | True | True | False | train_candidate |
| 2006 | train_2000_2010 | True | True | False | True | True | True | True | True | False | train_candidate |
| 2007 | train_2000_2010 | True | True | False | True | True | True | True | True | False | train_candidate |
| 2008 | train_2000_2010 | True | True | False | True | True | True | True | True | False | train_candidate |
| 2009 | train_2000_2010 | True | True | False | True | True | True | True | True | False | train_candidate |
| 2010 | train_2000_2010 | True | True | False | False | False | False | True | False | False | train_candidate |
| 2011 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2012 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2013 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2014 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2015 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2016 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2017 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2018 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2019 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2020 | test_2011_2020 | True | True | False | True | True | True | True | True | False | future_test_ready |
| 2021 | extension_2021_2023 | True | True | False | True | True | True | True | True | False | extension_test_ready |
| 2022 | extension_2021_2023 | True | True | False | True | True | True | True | True | False | extension_test_ready |
| 2023 | extension_2021_2023 | True | True | False | True | True | True | True | True | False | extension_test_ready |

## Missing DSSAT-auto years

No missing DSSAT-auto gaps among otherwise baseline-ready years.

## Split counts

| split | recommended_use | size |
| --- | --- | --- |
| extension_2021_2023 | extension_test_ready | 3 |
| test_2011_2020 | future_test_ready | 10 |
| train_2000_2010 | train_candidate | 6 |

## Recommendation

- Use all available 2000-2010 LC years as the historical training/development pool if the RL environment can instantiate them.
- Use 2011-2020 as future-year tests after DSSAT-auto gaps are completed.
- Treat 2021-2023 as external/extension tests.
- Complete missing DSSAT-auto years in a separate task before final five-scenario plotting.
