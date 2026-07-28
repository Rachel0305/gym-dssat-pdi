# 032_19 LC 75k daily stress plot data-quality audit record

## Status

- Completed.
- Training runs: 0.
- DSSAT reruns: 0.
- 032_17 figures were not modified in this audit.

## Key findings

| finding | evidence | risk |
| --- | --- | --- |
| rl_candidate_duplicate_date_dap_rows | 80 duplicate date-DAP rows; affected years=[2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020] | PPO weather totals and early daily line plots can be distorted if plotted without merging duplicate DAP/date rows. |
| baseline_daily_water_col_matches_plantgro_WSPD | max baseline water diff=0.023 | If large, 032_17 water panel is using an inconsistent water-stress source for baselines. |
| baseline_daily_n_col_matches_plantgro_NSTD | max baseline nitrogen diff=0.1551470332145691 | If large, 032_17 nitrogen panel is using an inconsistent nitrogen-stress source for baselines. |
| weather_total_should_be_scenario_invariant | max rain total spread across scenarios=159.59999999999997 mm | Weather is exogenous; nonzero spread usually indicates duplicated rows or date alignment differences, not real weather differences. |

## Duplicate date-DAP rows

| requested_year | scenario | date | dap | n_rows |
| --- | --- | --- | --- | --- |
| 2005 | rl_candidate | 2005-06-15 | 1.0 | 2 |
| 2005 | rl_candidate | 2005-06-16 | 2.0 | 2 |
| 2005 | rl_candidate | 2005-06-17 | 3.0 | 2 |
| 2005 | rl_candidate | 2005-06-18 | 4.0 | 2 |
| 2005 | rl_candidate | 2005-06-19 | 5.0 | 2 |
| 2006 | rl_candidate | 2006-06-15 | 1.0 | 2 |
| 2006 | rl_candidate | 2006-06-16 | 2.0 | 2 |
| 2006 | rl_candidate | 2006-06-17 | 3.0 | 2 |
| 2006 | rl_candidate | 2006-06-18 | 4.0 | 2 |
| 2006 | rl_candidate | 2006-06-19 | 5.0 | 2 |
| 2007 | rl_candidate | 2007-06-15 | 1.0 | 2 |
| 2007 | rl_candidate | 2007-06-16 | 2.0 | 2 |
| 2007 | rl_candidate | 2007-06-17 | 3.0 | 2 |
| 2007 | rl_candidate | 2007-06-18 | 4.0 | 2 |
| 2007 | rl_candidate | 2007-06-19 | 5.0 | 2 |
| 2008 | rl_candidate | 2008-06-14 | 1.0 | 2 |
| 2008 | rl_candidate | 2008-06-15 | 2.0 | 2 |
| 2008 | rl_candidate | 2008-06-16 | 3.0 | 2 |
| 2008 | rl_candidate | 2008-06-17 | 4.0 | 2 |
| 2008 | rl_candidate | 2008-06-18 | 5.0 | 2 |
| 2009 | rl_candidate | 2009-06-15 | 1.0 | 2 |
| 2009 | rl_candidate | 2009-06-16 | 2.0 | 2 |
| 2009 | rl_candidate | 2009-06-17 | 3.0 | 2 |
| 2009 | rl_candidate | 2009-06-18 | 4.0 | 2 |
| 2009 | rl_candidate | 2009-06-19 | 5.0 | 2 |
| 2010 | rl_candidate | 2010-06-15 | 1.0 | 2 |
| 2010 | rl_candidate | 2010-06-16 | 2.0 | 2 |
| 2010 | rl_candidate | 2010-06-17 | 3.0 | 2 |
| 2010 | rl_candidate | 2010-06-18 | 4.0 | 2 |
| 2010 | rl_candidate | 2010-06-19 | 5.0 | 2 |
| 2011 | rl_candidate | 2011-06-15 | 1.0 | 2 |
| 2011 | rl_candidate | 2011-06-16 | 2.0 | 2 |
| 2011 | rl_candidate | 2011-06-17 | 3.0 | 2 |
| 2011 | rl_candidate | 2011-06-18 | 4.0 | 2 |
| 2011 | rl_candidate | 2011-06-19 | 5.0 | 2 |
| 2012 | rl_candidate | 2012-06-14 | 1.0 | 2 |
| 2012 | rl_candidate | 2012-06-15 | 2.0 | 2 |
| 2012 | rl_candidate | 2012-06-16 | 3.0 | 2 |
| 2012 | rl_candidate | 2012-06-17 | 4.0 | 2 |
| 2012 | rl_candidate | 2012-06-18 | 5.0 | 2 |

## Rain total spread by year

| requested_year | min | max | rain_total_spread_mm |
| --- | --- | --- | --- |
| 2005 | 312.5 | 312.5 | 0.0 |
| 2006 | 347.2 | 347.2 | 0.0 |
| 2007 | 156.6 | 160.4 | 3.8 |
| 2008 | 378.2 | 417.1 | 38.9 |
| 2009 | 417.1 | 433.0 | 15.9 |
| 2010 | 97.8 | 257.4 | 159.6 |
| 2011 | 300.9 | 311.1 | 10.2 |
| 2012 | 262.0 | 262.0 | 0.0 |
| 2013 | 379.7 | 379.7 | 0.0 |
| 2014 | 196.4 | 204.1 | 7.7 |
| 2015 | 328.6 | 328.6 | 0.0 |
| 2016 | 278.0 | 298.5 | 20.5 |
| 2017 | 170.2 | 170.2 | 0.0 |
| 2018 | 194.7 | 194.7 | 0.0 |
| 2019 | 205.1 | 206.2 | 1.1 |
| 2020 | 254.3 | 254.3 | 0.0 |

## Baseline stress columns vs PlantGro WSPD/NSTD

| year | scenario | snapshot_available | daily_max_water_col | plantgro_max_WSPD | max_abs_diff_water_col_vs_WSPD | daily_max_n_col | plantgro_max_NSTD | max_abs_diff_n_col_vs_NSTD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | null | True | 0.0 | 0.0 | 0.0 | 0.490205 | 0.49 | 0.11647 |
| 2005 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.00816 | 0.012 | 0.008215 |
| 2005 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.490205 | 0.49 | 0.11647 |
| 2005 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.00816 | 0.012 | 0.008215 |
| 2006 | null | True | 0.0 | 0.0 | 0.0 | 0.503619 | 0.504 | 0.096249 |
| 2006 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.007002 | 0.012 | 0.007 |
| 2006 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.503619 | 0.504 | 0.096249 |
| 2006 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.007002 | 0.012 | 0.007 |
| 2007 | null | True | 0.0 | 0.0 | 0.0 | 0.455416 | 0.455 | 0.152631 |
| 2007 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2007 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.455416 | 0.455 | 0.152631 |
| 2007 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2008 | null | True | 0.0 | 0.0 | 0.0 | 0.480417 | 0.48 | 0.084387 |
| 2008 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2008 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.480417 | 0.48 | 0.084387 |
| 2008 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.023508 | 0.024 | 0.019976 |
| 2009 | null | True | 0.0 | 0.0 | 0.0 | 0.479579 | 0.48 | 0.086934 |
| 2009 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2009 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.479579 | 0.48 | 0.086934 |
| 2009 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2010 | null | False |  |  |  |  |  |  |
| 2010 | recorded_farmer | False |  |  |  |  |  |  |
| 2010 | dssat_auto | False |  |  |  |  |  |  |
| 2010 | official_extension_expert | False |  |  |  |  |  |  |
| 2011 | null | True | 0.0 | 0.0 | 0.0 | 0.483917 | 0.484 | 0.127929 |
| 2011 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.014754 | 0.015 | 0.015 |
| 2011 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.483917 | 0.484 | 0.127929 |
| 2011 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.023377 | 0.023 | 0.023377 |
| 2012 | null | True | 0.0 | 0.0 | 0.0 | 0.472697 | 0.473 | 0.111524 |
| 2012 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2012 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.472697 | 0.473 | 0.111524 |
| 2012 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2013 | null | True | 0.0 | 0.0 | 0.0 | 0.494373 | 0.494 | 0.105594 |
| 2013 | recorded_farmer | True | 0.022514 | 0.023 | 0.023 | 0.012191 | 0.012 | 0.012191 |
| 2013 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.494373 | 0.494 | 0.105594 |
| 2013 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2014 | null | True | 0.0 | 0.0 | 0.0 | 0.486402 | 0.486 | 0.094353 |
| 2014 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2014 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.486402 | 0.486 | 0.094353 |
| 2014 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.023986 | 0.024 | 0.02161 |
| 2015 | null | True | 0.0 | 0.0 | 0.0 | 0.485397 | 0.485 | 0.103182 |
| 2015 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.011634 | 0.012 | 0.010299 |
| 2015 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.485397 | 0.485 | 0.103182 |
| 2015 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.011634 | 0.012 | 0.010299 |
| 2016 | null | True | 0.0 | 0.0 | 0.0 | 0.461121 | 0.461 | 0.101089 |
| 2016 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.024556 | 0.025 | 0.025 |
| 2016 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.461121 | 0.461 | 0.101089 |
| 2016 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.006725 | 0.012 | 0.007 |
| 2017 | null | True | 0.0 | 0.0 | 0.0 | 0.488146 | 0.488 | 0.115498 |
| 2017 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.006021 | 0.012 | 0.005979 |
| 2017 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.488146 | 0.488 | 0.115498 |
| 2017 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.00766 | 0.012 | 0.008 |
| 2018 | null | True | 0.0 | 0.0 | 0.0 | 0.477189 | 0.477 | 0.155147 |
| 2018 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.007682 | 0.012 | 0.008 |
| 2018 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.477189 | 0.477 | 0.155147 |
| 2018 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.007682 | 0.012 | 0.008 |
| 2019 | null | True | 0.0 | 0.0 | 0.0 | 0.503734 | 0.504 | 0.112868 |
| 2019 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.005445 | 0.012 | 0.006555 |
| 2019 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.503734 | 0.504 | 0.112868 |
| 2019 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.046266 | 0.046 | 0.036513 |
| 2020 | null | True | 0.0 | 0.0 | 0.0 | 0.49011 | 0.49 | 0.108726 |
| 2020 | recorded_farmer | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |
| 2020 | dssat_auto | True | 0.0 | 0.0 | 0.0 | 0.49011 | 0.49 | 0.108726 |
| 2020 | official_extension_expert | True | 0.0 | 0.0 | 0.0 | 0.012191 | 0.012 | 0.012191 |

## Interpretation

- `WSPD/NSTD` panels should be interpreted from PlantGro-derived columns when available.
- `rl_candidate` rows in 032_17 require date-DAP de-duplication before weather totals or daily stress panels are advisor-facing.
- LC years with low rainfall but null WSPD=0 are not automatically erroneous: DSSAT PlantGro may report no water stress while nitrogen stress is severe.
- The next fix should rebuild daily figures from consistent sources, preferably PlantGro/MgmtEvent snapshots for all scenarios, or at least de-duplicate PPO rows and relabel environment-derived `swfac/nstres` cautiously.
