# 052_02 LCA/LC originIC four-baseline static level-1 record

- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`.
- Source MZX: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/LC/CNLC0801.MZX`.
- Source MZX SHA256: `de0d5deddbb7c7a93d4576488fdef0c120ac11d49a2c611cd83699fb4a24b77b`.
- Years: `[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]`.
- Elapsed: `115.5` s.
- Static recorded/expert rows were rendered with DSSAT management level ID `1` using the 037_07 correction.
- Recorded farmer template remains frozen/static; it is not tuned for PPO and does not represent each year's true farmer management.
- Recorded template totals: `130.0` mm irrigation / `250.0` kg/ha nitrogen.
- Summary CSV: `benchmark_results/052_02_lca_originIC_four_baselines_static_level1/evaluation/052_02_baseline_summary.csv`.
- Event audit CSV: `benchmark_results/052_02_lca_originIC_four_baselines_static_level1/evaluation/052_02_management_event_audit.csv`.

## Scenario means

| scenario | years | mean_grain_yield_kg_ha | mean_biomass_kg_ha | mean_irrigation_mm | mean_nitrogen_kg_ha | mean_WP_ET_kg_m3 | mean_PFP_N_kg_kg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dssat_auto | 10 | 9335.5541 | 18273.2196 | 74.9 | 0.0 | 2.763 |  |
| null | 10 | 9318.0552 | 18271.7196 | 0.0 | 0.0 | 2.876 |  |
| official_extension_expert | 10 | 9354.6377 | 18183.731 | 199.0 | 245.0 | 2.772 | 38.17 |
| recorded_farmer_template | 10 | 9325.6194 | 18153.0251 | 130.0 | 250.0 | 2.757 | 37.3 |

## Non-ok manifest rows

_No rows._

## Event-chain audit preview

| year | scenario | planned_i_events | planned_i_total | mgmt_i_events | mgmt_i_total | planned_n_events | planned_n_total | mgmt_n_events | mgmt_n_total | status | issues | planned_vs_inp_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2014 | dssat_auto | 0 | 0.0 | 2 | 87.8 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2015 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2015 | dssat_auto | 0 | 0.0 | 2 | 87.3 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2016 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2016 | dssat_auto | 0 | 0.0 | 1 | 45.5 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2017 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2017 | dssat_auto | 0 | 0.0 | 2 | 85.3 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2018 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2018 | dssat_auto | 0 | 0.0 | 3 | 130.7 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2019 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2019 | dssat_auto | 0 | 0.0 | 2 | 88.9 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2020 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2020 | dssat_auto | 0 | 0.0 | 2 | 88.6 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2021 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2021 | dssat_auto | 0 | 0.0 | 1 | 45.9 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2022 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2022 | dssat_auto | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2023 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | recorded_farmer_template | 2 | 130.0 | 2 | 130.0 | 2 | 250.0 | 2 | 250.0 | ok |  |  |
| 2023 | dssat_auto | 0 | 0.0 | 2 | 88.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
