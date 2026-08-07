# 051_02 FQA/FQ originIC four-baseline static level-1 record

- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`.
- Source MZX: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0801.MZX`.
- Source MZX SHA256: `3bfe509af06d2c724b0635d0150ba31544d338039634ba6cc98b64d1c8422135`.
- Years: `[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]`.
- Elapsed: `113.2` s.
- Static recorded/expert rows were rendered with DSSAT management level ID `1` using the 037_07 correction.
- Recorded farmer template remains frozen/static; it is not tuned for PPO and does not represent each year's true farmer management.
- Recorded template totals: `75.0` mm irrigation / `144.0` kg/ha nitrogen.
- Summary CSV: `benchmark_results/051_02_fqa_originIC_four_baselines_static_level1/evaluation/051_02_baseline_summary.csv`.
- Event audit CSV: `benchmark_results/051_02_fqa_originIC_four_baselines_static_level1/evaluation/051_02_management_event_audit.csv`.

## Scenario means

| scenario | years | mean_grain_yield_kg_ha | mean_biomass_kg_ha | mean_irrigation_mm | mean_nitrogen_kg_ha | mean_WP_ET_kg_m3 | mean_PFP_N_kg_kg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dssat_auto | 10 | 7178.1609 | 13182.2281 | 34.7 | 0.0 | 2.125 |  |
| null | 10 | 6953.2554 | 12959.4002 | 0.0 | 0.0 | 2.145 |  |
| official_extension_expert | 10 | 7312.007 | 12901.1587 | 212.9 | 241.7 | 2.079 | 33.1556 |
| recorded_farmer_template | 10 | 7201.7732 | 13155.0179 | 75.0 | 144.0 | 2.145 | 55.5778 |

## Non-ok manifest rows

_No rows._

## Event-chain audit preview

| year | scenario | planned_i_events | planned_i_total | mgmt_i_events | mgmt_i_total | planned_n_events | planned_n_total | mgmt_n_events | mgmt_n_total | status | issues | planned_vs_inp_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2014 | dssat_auto | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | official_extension_expert | 6 | 228.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2015 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2015 | dssat_auto | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | official_extension_expert | 6 | 228.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2016 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2016 | dssat_auto | 0 | 0.0 | 3 | 57.5 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2017 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2017 | dssat_auto | 0 | 0.0 | 2 | 40.2 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | official_extension_expert | 5 | 198.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2018 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2018 | dssat_auto | 0 | 0.0 | 4 | 76.1 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | official_extension_expert | 4 | 157.5 | 4 | 157.6 | 4 | 213.75 | 4 | 212.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_event_count_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2019 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2019 | dssat_auto | 0 | 0.0 | 6 | 114.8 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | official_extension_expert | 6 | 228.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2020 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2020 | dssat_auto | 0 | 0.0 | 1 | 20.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | official_extension_expert | 6 | 228.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2021 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2021 | dssat_auto | 0 | 0.0 | 1 | 19.2 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | official_extension_expert | 6 | 228.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2022 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2022 | dssat_auto | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2023 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | recorded_farmer_template | 1 | 75.0 | 1 | 75.0 | 1 | 144.0 | 1 | 144.0 | ok |  |  |
| 2023 | dssat_auto | 0 | 0.0 | 3 | 58.5 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
