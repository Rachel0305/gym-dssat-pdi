# 031_29 SY all-year baseline completion record

## Scope

- No PPO/DQN training was run.
- 031_27 frozen PPO candidates were not changed.
- Generated baselines use the same SYA all-year rendered DSSAT environment as 031_27.
- True recorded-farmer management is not reconstructed where original management evidence is unavailable.
- Recorded-template scenarios, if present, are counterfactual transfers of the 2012/2014/2015 recorded schedules.

## Coverage summary

| scenario | coverage_status | n |
| --- | --- | --- |
| dssat_auto | generated_031_29 | 19 |
| null | generated_031_29 | 19 |
| official_extension_expert | generated_031_29 | 19 |
| recorded_farmer_template_2012 | generated_031_29 | 19 |
| recorded_farmer_template_2014 | generated_031_29 | 19 |
| recorded_farmer_template_2015 | generated_031_29 | 19 |

## Generated baseline metric preview

| year | scenario | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | dssat_auto | 2617.0 | 0.0 | 0.0 | 0.64 |  | 0.345 | 0.4477 |
| 2005 | null | 2617.0 | 0.0 | 0.0 | 0.64 |  | 0.345 | 0.4477 |
| 2005 | official_extension_expert | 9186.0 | 266.0 | 300.0 | 1.99 | 30.6 | 0.345 | 0.0164 |
| 2005 | recorded_farmer_template_2012 | 9667.0 | 0.0 | 247.0 | 2.1 | 39.1 | 0.345 | 0.0164 |
| 2005 | recorded_farmer_template_2014 | 9666.0 | 0.0 | 247.0 | 2.1 | 39.1 | 0.345 | 0.0164 |
| 2005 | recorded_farmer_template_2015 | 9666.0 | 0.0 | 247.0 | 2.1 | 39.1 | 0.345 | 0.0164 |
| 2006 | dssat_auto | 2633.0 | 0.0 | 0.0 | 0.74 |  | 0.345 | 0.4335 |
| 2006 | null | 2633.0 | 0.0 | 0.0 | 0.74 |  | 0.345 | 0.4335 |
| 2006 | official_extension_expert | 10178.0 | 266.0 | 300.0 | 2.22 | 33.9 | 0.345 | 0.0133 |
| 2006 | recorded_farmer_template_2012 | 10388.0 | 0.0 | 247.0 | 2.34 | 42.1 | 0.345 | 0.0133 |
| 2006 | recorded_farmer_template_2014 | 10388.0 | 0.0 | 247.0 | 2.34 | 42.1 | 0.345 | 0.0133 |
| 2006 | recorded_farmer_template_2015 | 10388.0 | 0.0 | 247.0 | 2.34 | 42.1 | 0.345 | 0.0133 |
| 2007 | dssat_auto | 2621.0 | 33.0 | 0.0 | 0.81 |  | 0.345 | 0.4607 |
| 2007 | null | 2528.0 | 0.0 | 0.0 | 0.82 |  | 0.345 | 0.4766 |
| 2007 | official_extension_expert | 11282.0 | 266.0 | 300.0 | 2.61 | 37.6 | 0.345 | 0.016 |
| 2007 | recorded_farmer_template_2012 | 10825.0 | 0.0 | 247.0 | 2.66 | 43.8 | 0.345 | 0.076 |
| 2007 | recorded_farmer_template_2014 | 10811.0 | 0.0 | 247.0 | 2.66 | 43.8 | 0.345 | 0.09 |
| 2007 | recorded_farmer_template_2015 | 10815.0 | 0.0 | 247.0 | 2.66 | 43.8 | 0.345 | 0.0811 |
| 2008 | dssat_auto | 3016.0 | 0.0 | 0.0 | 0.76 |  | 0.345 | 0.4075 |
| 2008 | null | 3016.0 | 0.0 | 0.0 | 0.76 |  | 0.345 | 0.4075 |
| 2008 | official_extension_expert | 10281.0 | 266.0 | 300.0 | 2.12 | 34.3 | 0.345 | 0.0156 |
| 2008 | recorded_farmer_template_2012 | 11145.0 | 0.0 | 247.0 | 2.38 | 45.1 | 0.345 | 0.0156 |
| 2008 | recorded_farmer_template_2014 | 11145.0 | 0.0 | 247.0 | 2.38 | 45.1 | 0.345 | 0.0156 |
| 2008 | recorded_farmer_template_2015 | 11145.0 | 0.0 | 247.0 | 2.38 | 45.1 | 0.345 | 0.0156 |
| 2009 | dssat_auto | 2848.0 | 33.0 | 0.0 | 0.76 |  | 0.345 | 0.4255 |
| 2009 | null | 2800.0 | 0.0 | 0.0 | 0.77 |  | 0.345 | 0.4255 |
| 2009 | official_extension_expert | 12355.0 | 266.0 | 300.0 | 2.52 | 41.2 | 0.345 | 0.0122 |
| 2009 | recorded_farmer_template_2012 | 11911.0 | 0.0 | 247.0 | 2.61 | 48.2 | 0.5733 | 0.1019 |
| 2009 | recorded_farmer_template_2014 | 11831.0 | 0.0 | 247.0 | 2.61 | 47.9 | 0.6032 | 0.0807 |
| 2009 | recorded_farmer_template_2015 | 11851.0 | 0.0 | 247.0 | 2.61 | 48.0 | 0.5956 | 0.0836 |
| 2010 | dssat_auto | 2359.0 | 0.0 | 0.0 | 0.63 |  | 0.345 | 0.4449 |
| 2010 | null | 2359.0 | 0.0 | 0.0 | 0.63 |  | 0.345 | 0.4449 |
| 2010 | official_extension_expert | 8181.0 | 266.0 | 300.0 | 1.8 | 27.3 | 0.345 | 0.0166 |
| 2010 | recorded_farmer_template_2012 | 8503.0 | 0.0 | 247.0 | 1.99 | 34.4 | 0.345 | 0.0153 |
| 2010 | recorded_farmer_template_2014 | 8504.0 | 0.0 | 247.0 | 1.99 | 34.4 | 0.345 | 0.0153 |
| 2010 | recorded_farmer_template_2015 | 8504.0 | 0.0 | 247.0 | 1.99 | 34.4 | 0.345 | 0.0153 |
| 2011 | dssat_auto | 2559.0 | 0.0 | 0.0 | 0.66 |  | 0.345 | 0.4361 |
| 2011 | null | 2559.0 | 0.0 | 0.0 | 0.66 |  | 0.345 | 0.4361 |
| 2011 | official_extension_expert | 9282.0 | 266.0 | 300.0 | 2.01 | 30.9 | 0.345 | 0.0161 |
| 2011 | recorded_farmer_template_2012 | 9553.0 | 0.0 | 247.0 | 2.14 | 38.7 | 0.345 | 0.0161 |
| 2011 | recorded_farmer_template_2014 | 9553.0 | 0.0 | 247.0 | 2.14 | 38.7 | 0.345 | 0.0161 |
| 2011 | recorded_farmer_template_2015 | 9553.0 | 0.0 | 247.0 | 2.14 | 38.7 | 0.345 | 0.0161 |
| 2012 | dssat_auto | 2644.0 | 0.0 | 0.0 | 0.73 |  | 0.345 | 0.415 |
| 2012 | null | 2644.0 | 0.0 | 0.0 | 0.73 |  | 0.345 | 0.415 |
| 2012 | official_extension_expert | 9118.0 | 266.0 | 300.0 | 2.0 | 30.4 | 0.345 | 0.015 |
| 2012 | recorded_farmer_template_2012 | 10167.0 | 0.0 | 247.0 | 2.29 | 41.2 | 0.345 | 0.015 |
| 2012 | recorded_farmer_template_2014 | 10167.0 | 0.0 | 247.0 | 2.29 | 41.2 | 0.345 | 0.015 |
| 2012 | recorded_farmer_template_2015 | 10167.0 | 0.0 | 247.0 | 2.29 | 41.2 | 0.345 | 0.015 |
| 2013 | dssat_auto | 2475.0 | 33.0 | 0.0 | 0.66 |  | 0.345 | 0.4316 |
| 2013 | null | 2475.0 | 0.0 | 0.0 | 0.67 |  | 0.345 | 0.4271 |
| 2013 | official_extension_expert | 10603.0 | 266.0 | 300.0 | 2.34 | 35.3 | 0.345 | 0.0136 |
| 2013 | recorded_farmer_template_2012 | 10706.0 | 0.0 | 247.0 | 2.47 | 43.3 | 0.345 | 0.0136 |
| 2013 | recorded_farmer_template_2014 | 10704.0 | 0.0 | 247.0 | 2.47 | 43.3 | 0.345 | 0.0136 |
| 2013 | recorded_farmer_template_2015 | 10704.0 | 0.0 | 247.0 | 2.47 | 43.3 | 0.345 | 0.0136 |
| 2014 | dssat_auto | 2730.0 | 0.0 | 0.0 | 0.74 |  | 0.345 | 0.4378 |
| 2014 | null | 2730.0 | 0.0 | 0.0 | 0.74 |  | 0.345 | 0.4378 |
| 2014 | official_extension_expert | 10800.0 | 266.0 | 300.0 | 2.17 | 36.0 | 0.345 | 0.0122 |
| 2014 | recorded_farmer_template_2012 | 9490.0 | 0.0 | 247.0 | 2.09 | 38.4 | 0.7708 | 0.0122 |
| 2014 | recorded_farmer_template_2014 | 9380.0 | 0.0 | 247.0 | 2.08 | 38.0 | 0.8074 | 0.0122 |
| 2014 | recorded_farmer_template_2015 | 9410.0 | 0.0 | 247.0 | 2.09 | 38.1 | 0.7773 | 0.0122 |
| 2015 | dssat_auto | 2635.0 | 0.0 | 0.0 | 0.73 |  | 0.345 | 0.4211 |
| 2015 | null | 2635.0 | 0.0 | 0.0 | 0.73 |  | 0.345 | 0.4211 |
| 2015 | official_extension_expert | 10575.0 | 266.0 | 300.0 | 2.25 | 35.2 | 0.345 | 0.0122 |
| 2015 | recorded_farmer_template_2012 | 11177.0 | 0.0 | 247.0 | 2.44 | 45.3 | 0.345 | 0.0122 |
| 2015 | recorded_farmer_template_2014 | 11177.0 | 0.0 | 247.0 | 2.45 | 45.3 | 0.345 | 0.0122 |
| 2015 | recorded_farmer_template_2015 | 11177.0 | 0.0 | 247.0 | 2.45 | 45.3 | 0.345 | 0.0122 |
| 2016 | dssat_auto | 2161.0 | 0.0 | 0.0 | 0.6 |  | 0.345 | 0.4278 |
| 2016 | null | 2161.0 | 0.0 | 0.0 | 0.6 |  | 0.345 | 0.4278 |
| 2016 | official_extension_expert | 7781.0 | 266.0 | 300.0 | 1.65 | 25.9 | 0.345 | 0.0152 |
| 2016 | recorded_farmer_template_2012 | 8033.0 | 0.0 | 247.0 | 1.76 | 32.5 | 0.345 | 0.0152 |
| 2016 | recorded_farmer_template_2014 | 8034.0 | 0.0 | 247.0 | 1.76 | 32.5 | 0.345 | 0.0152 |
| 2016 | recorded_farmer_template_2015 | 8034.0 | 0.0 | 247.0 | 1.76 | 32.5 | 0.345 | 0.0152 |
| 2017 | dssat_auto | 2568.0 | 67.0 | 0.0 | 0.76 |  | 0.345 | 0.4528 |
| 2017 | null | 2294.0 | 0.0 | 0.0 | 0.75 |  | 0.345 | 0.4694 |
| 2017 | official_extension_expert | 10855.0 | 266.0 | 300.0 | 2.49 | 36.2 | 0.345 | 0.0144 |
| 2017 | recorded_farmer_template_2012 | 8953.0 | 0.0 | 247.0 | 2.39 | 36.2 | 0.8238 | 0.1538 |
| 2017 | recorded_farmer_template_2014 | 8965.0 | 0.0 | 247.0 | 2.39 | 36.3 | 0.8235 | 0.1538 |
| 2017 | recorded_farmer_template_2015 | 8962.0 | 0.0 | 247.0 | 2.39 | 36.3 | 0.8236 | 0.1538 |
| 2018 | dssat_auto | 2034.0 | 0.0 | 0.0 | 0.62 |  | 0.345 | 0.4298 |
| 2018 | null | 2034.0 | 0.0 | 0.0 | 0.62 |  | 0.345 | 0.4298 |

## Interpretation boundary

031_29 generated the requested scenarios listed in the manifest. Any `recorded_farmer_template_*` scenario is a transferred historical-management template, not a real recorded-farmer observation for the target year.