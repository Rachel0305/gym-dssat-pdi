# 055_02 YCA/YC lowIC four-baseline static level-1 record

- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`.
- Source MZX: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC0801.MZX`.
- Source MZX SHA256: `5070b63b92b953ce72cf2bb5734927b050d262986f72f9926809e73d0b53023d`.
- Years: `[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]`.
- Elapsed: `121.0` s.
- Static recorded/expert rows were rendered with DSSAT management level ID `1` using the 037_07 correction.
- Recorded farmer template remains frozen/static; it is not tuned for PPO and does not represent each year's true farmer management.
- Recorded template totals: `120.0` mm irrigation / `374.0` kg/ha nitrogen.
- Summary CSV: `benchmark_results/055_02_yca_lowIC_four_baselines_static_level1/evaluation/055_02_baseline_summary.csv`.
- Event audit CSV: `benchmark_results/055_02_yca_lowIC_four_baselines_static_level1/evaluation/055_02_management_event_audit.csv`.

## Scenario means

| scenario | years | mean_grain_yield_kg_ha | mean_biomass_kg_ha | mean_irrigation_mm | mean_nitrogen_kg_ha | mean_WP_ET_kg_m3 | mean_PFP_N_kg_kg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dssat_auto | 10 | 4417.1143 | 12414.3369 | 146.5 | 0.0 | 1.223 |  |
| null | 10 | 3290.8762 | 10112.8673 | 0.0 | 0.0 | 1.14 |  |
| official_extension_expert | 10 | 8200.5546 | 18952.6702 | 211.0 | 245.0 | 2.293 | 33.48 |
| recorded_farmer_template | 10 | 7972.7303 | 18131.7948 | 120.0 | 374.0 | 2.338 | 21.32 |

## Non-ok manifest rows

_No rows._

## Event-chain audit preview

| year | scenario | planned_i_events | planned_i_total | mgmt_i_events | mgmt_i_total | planned_n_events | planned_n_total | mgmt_n_events | mgmt_n_total | status | issues | planned_vs_inp_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2014 | dssat_auto | 0 | 0.0 | 4 | 202.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | official_extension_expert | 6 | 228.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2015 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2015 | dssat_auto | 0 | 0.0 | 4 | 196.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | official_extension_expert | 6 | 228.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2016 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2016 | dssat_auto | 0 | 0.0 | 4 | 174.7 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2017 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2017 | dssat_auto | 0 | 0.0 | 2 | 110.3 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2018 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2018 | dssat_auto | 0 | 0.0 | 3 | 155.6 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2019 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2019 | dssat_auto | 0 | 0.0 | 5 | 242.4 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2020 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2020 | dssat_auto | 0 | 0.0 | 2 | 110.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | official_extension_expert | 5 | 198.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2021 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2021 | dssat_auto | 0 | 0.0 | 1 | 57.8 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | official_extension_expert | 5 | 198.75 | 6 | 228.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2022 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2022 | dssat_auto | 0 | 0.0 | 1 | 58.9 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
| 2023 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | recorded_farmer_template | 1 | 120.0 | 1 | 120.0 | 2 | 374.0 | 2 | 374.0 | ok |  |  |
| 2023 | dssat_auto | 0 | 0.0 | 3 | 157.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | official_extension_expert | 5 | 198.75 | 5 | 198.8 | 5 | 247.5 | 5 | 245.0 | ok |  | irrigation_event_count_planned_vs_inp;irrigation_total_planned_vs_inp;nitrogen_total_planned_vs_inp |
