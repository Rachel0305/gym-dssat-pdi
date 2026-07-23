# 031_28 SY all-year four-baseline completion record

## Scope

- No PPO/DQN training was run.
- 031_27 frozen PPO candidates were not changed.
- Generated baselines use the same SYA all-year rendered DSSAT environment as 031_27.
- Recorded-farmer management is not reconstructed where original management evidence is unavailable.

## Coverage summary

| scenario | coverage_status | n |
| --- | --- | --- |
| dssat_auto | blocked_auto_management_fortran_error_in_all_year_template | 19 |
| null | generated_031_28 | 19 |
| official_extension_expert | generated_031_28 | 19 |
| recorded_farmer | blocked_missing_recorded_management_source | 16 |
| recorded_farmer | reused_028_02 | 3 |

## Generated baseline metric preview

| year | scenario | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | null | 2617.0 | 0.0 | 0.0 | 0.64 |  | 0.345 | 0.4477 |
| 2005 | official_extension_expert | 9186.0 | 266.0 | 300.0 | 1.99 | 30.6 | 0.345 | 0.0164 |
| 2006 | null | 2633.0 | 0.0 | 0.0 | 0.74 |  | 0.345 | 0.4335 |
| 2006 | official_extension_expert | 10178.0 | 266.0 | 300.0 | 2.22 | 33.9 | 0.345 | 0.0133 |
| 2007 | null | 2528.0 | 0.0 | 0.0 | 0.82 |  | 0.345 | 0.4766 |
| 2007 | official_extension_expert | 11282.0 | 266.0 | 300.0 | 2.61 | 37.6 | 0.345 | 0.016 |
| 2008 | null | 3016.0 | 0.0 | 0.0 | 0.76 |  | 0.345 | 0.4075 |
| 2008 | official_extension_expert | 10281.0 | 266.0 | 300.0 | 2.12 | 34.3 | 0.345 | 0.0156 |
| 2009 | null | 2800.0 | 0.0 | 0.0 | 0.77 |  | 0.345 | 0.4255 |
| 2009 | official_extension_expert | 12355.0 | 266.0 | 300.0 | 2.52 | 41.2 | 0.345 | 0.0122 |
| 2010 | null | 2359.0 | 0.0 | 0.0 | 0.63 |  | 0.345 | 0.4449 |
| 2010 | official_extension_expert | 8181.0 | 266.0 | 300.0 | 1.8 | 27.3 | 0.345 | 0.0166 |
| 2011 | null | 2559.0 | 0.0 | 0.0 | 0.66 |  | 0.345 | 0.4361 |
| 2011 | official_extension_expert | 9282.0 | 266.0 | 300.0 | 2.01 | 30.9 | 0.345 | 0.0161 |
| 2012 | null | 2644.0 | 0.0 | 0.0 | 0.73 |  | 0.345 | 0.415 |
| 2012 | official_extension_expert | 9118.0 | 266.0 | 300.0 | 2.0 | 30.4 | 0.345 | 0.015 |
| 2013 | null | 2475.0 | 0.0 | 0.0 | 0.67 |  | 0.345 | 0.4271 |
| 2013 | official_extension_expert | 10603.0 | 266.0 | 300.0 | 2.34 | 35.3 | 0.345 | 0.0136 |
| 2014 | null | 2730.0 | 0.0 | 0.0 | 0.74 |  | 0.345 | 0.4378 |
| 2014 | official_extension_expert | 10800.0 | 266.0 | 300.0 | 2.17 | 36.0 | 0.345 | 0.0122 |
| 2015 | null | 2635.0 | 0.0 | 0.0 | 0.73 |  | 0.345 | 0.4211 |
| 2015 | official_extension_expert | 10575.0 | 266.0 | 300.0 | 2.25 | 35.2 | 0.345 | 0.0122 |
| 2016 | null | 2161.0 | 0.0 | 0.0 | 0.6 |  | 0.345 | 0.4278 |
| 2016 | official_extension_expert | 7781.0 | 266.0 | 300.0 | 1.65 | 25.9 | 0.345 | 0.0152 |
| 2017 | null | 2294.0 | 0.0 | 0.0 | 0.75 |  | 0.345 | 0.4694 |
| 2017 | official_extension_expert | 10855.0 | 266.0 | 300.0 | 2.49 | 36.2 | 0.345 | 0.0144 |
| 2018 | null | 2034.0 | 0.0 | 0.0 | 0.62 |  | 0.345 | 0.4298 |
| 2018 | official_extension_expert | 8039.0 | 266.0 | 300.0 | 1.82 | 26.8 | 0.345 | 0.017 |
| 2019 | null | 2300.0 | 0.0 | 0.0 | 0.59 |  | 0.345 | 0.4575 |
| 2019 | official_extension_expert | 10155.0 | 266.0 | 300.0 | 2.11 | 33.8 | 0.345 | 0.013 |
| 2020 | null | 2416.0 | 0.0 | 0.0 | 0.67 |  | 0.345 | 0.4478 |
| 2020 | official_extension_expert | 9825.0 | 266.0 | 300.0 | 2.1 | 32.8 | 0.345 | 0.0162 |
| 2021 | null | 2709.0 | 0.0 | 0.0 | 0.72 |  | 0.345 | 0.4473 |
| 2021 | official_extension_expert | 9892.0 | 266.0 | 300.0 | 2.15 | 33.0 | 0.345 | 0.0136 |
| 2022 | null | 2551.0 | 0.0 | 0.0 | 0.67 |  | 0.345 | 0.4416 |
| 2022 | official_extension_expert | 10715.0 | 266.0 | 300.0 | 2.37 | 35.7 | 0.345 | 0.0122 |
| 2023 | null | 2233.0 | 0.0 | 0.0 | 0.6 |  | 0.345 | 0.4717 |
| 2023 | official_extension_expert | 10877.0 | 266.0 | 300.0 | 2.3 | 36.3 | 0.345 | 0.016 |

## Interpretation boundary

031_28 completes reproducible generated `null` and `official_extension_expert` baselines for available SYA years. `dssat_auto` is marked unavailable for this all-year rendered template after a smoke-test DSSAT automatic-management error, and recorded-farmer baselines are not created for years without original recorded management source rows.