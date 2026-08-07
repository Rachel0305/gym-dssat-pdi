# 054_02 HLA lowIC four-baseline static level-1 record

- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`.
- Source MZX: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL/CNHL0701_corrected_IC123.MZX`.
- Source MZX SHA256: `384937c36487217bca7a33bf5e64d1de3d9e8a841408f638e77cad71b06c0a04`.
- Years: `[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]`.
- Elapsed: `161.6` s.
- Static recorded/expert rows were rendered with DSSAT management level ID `1` using the 037_07 correction.
- Recorded farmer template remains frozen/static; it is not tuned for PPO and does not represent each year's true farmer management.
- Recorded template totals: `30.0` mm irrigation / `165.0` kg/ha nitrogen.
- Summary CSV: `benchmark_results/054_02_hla_lowIC_four_baselines_static_level1/evaluation/054_02_baseline_summary.csv`.
- Event audit CSV: `benchmark_results/054_02_hla_lowIC_four_baselines_static_level1/evaluation/054_02_management_event_audit.csv`.

## Scenario means

| scenario | years | mean_grain_yield_kg_ha | mean_biomass_kg_ha | mean_irrigation_mm | mean_nitrogen_kg_ha | mean_WP_ET_kg_m3 | mean_PFP_N_kg_kg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dssat_auto | 10 | 6027.5218 | 14615.4581 | 164.0 | 0.0 | 1.341 |  |
| null | 10 | 4653.4852 | 11699.5645 | 0.0 | 0.0 | 1.173 |  |
| official_extension_expert | 10 | 6796.8848 | 16381.9607 | 266.0 | 297.0 | 1.475 | 22.89 |
| recorded_farmer_template | 10 | 5482.2989 | 14090.8824 | 30.0 | 165.0 | 1.341 | 33.22 |

## Non-ok manifest rows

_No rows._

## Event-chain audit preview

| year | scenario | planned_i_events | planned_i_total | mgmt_i_events | mgmt_i_total | planned_n_events | planned_n_total | mgmt_n_events | mgmt_n_total | status | issues | planned_vs_inp_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2014 | dssat_auto | 0 | 0.0 | 2 | 120.2 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2014 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2015 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2015 | dssat_auto | 0 | 0.0 | 5 | 269.6 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2015 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2016 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2016 | dssat_auto | 0 | 0.0 | 4 | 224.7 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2016 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2017 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2017 | dssat_auto | 0 | 0.0 | 3 | 174.4 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2017 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2018 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2018 | dssat_auto | 0 | 0.0 | 2 | 125.7 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2018 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2019 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2019 | dssat_auto | 0 | 0.0 | 2 | 125.5 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2019 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2020 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2020 | dssat_auto | 0 | 0.0 | 2 | 125.9 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2020 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2021 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2021 | dssat_auto | 0 | 0.0 | 1 | 77.1 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2021 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2022 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2022 | dssat_auto | 0 | 0.0 | 4 | 222.2 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2022 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
| 2023 | null | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | recorded_farmer_template | 3 | 30.0 | 3 | 30.0 | 1 | 165.0 | 1 | 165.0 | ok |  |  |
| 2023 | dssat_auto | 0 | 0.0 | 3 | 173.9 | 0 | 0.0 | 0 | 0.0 | ok |  |  |
| 2023 | official_extension_expert | 8 | 266.25 | 8 | 266.1 | 6 | 300.0 | 6 | 297.0 | ok |  | nitrogen_total_planned_vs_inp |
