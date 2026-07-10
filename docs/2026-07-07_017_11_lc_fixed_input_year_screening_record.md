# 017_11 LC fixed-input yearly baseline screening

## Purpose

Screen LC 2008-2011 after temporary input fixes. This stage does not train DQN and does not modify the source LC input package.

## Temporary fixes

- Soil ID: `LC99001200` -> `LC990012007`.
- IC levels were cloned by treatment year: 08121, 09121, 10121, 11121.
- Treatment IC pointers and simulation SDATE were aligned to the cloned IC levels.

## Run status

| year | scenario | returncode | timed_out | run_dir |
| --- | --- | --- | --- | --- |
| 2008 | null | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2008/null |
| 2008 | recorded | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2008/recorded |
| 2008 | dssat_auto | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2008/dssat_auto |
| 2009 | null | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2009/null |
| 2009 | recorded | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2009/recorded |
| 2009 | dssat_auto | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2009/dssat_auto |
| 2010 | null | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2010/null |
| 2010 | recorded | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2010/recorded |
| 2010 | dssat_auto | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2010/dssat_auto |
| 2011 | null | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2011/null |
| 2011 | recorded | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2011/recorded |
| 2011 | dssat_auto | 0 | False | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2011/dssat_auto |

## Summary

| year | scenario | final_gwad | final_cwad | event_irrigation_total | event_fertilizer_total | max_water_stress | max_nitrogen_stress | yield_gain_vs_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2008 | null | 9253.0 | 17866.0 | 0.0 | 0.0 | 0.0 | 0.012 | 0.0 |
| 2008 | recorded | 9253.0 | 17853.0 | 60.0 | 276.0 | 0.0 | 0.012 | 0.0 |
| 2008 | dssat_auto | 9253.0 | 17866.0 | 0.0 | 0.0 | 0.0 | 0.012 | 0.0 |
| 2009 | null | 9089.0 | 17989.0 | 0.0 | 0.0 | 0.0 | 0.012 | 0.0 |
| 2009 | recorded | 9128.0 | 18060.0 | 65.0 | 750.0 | 0.0 | 0.012 | 39.0 |
| 2009 | dssat_auto | 9128.0 | 18056.0 | 90.9 | 0.0 | 0.0 | 0.012 | 39.0 |
| 2010 | null | 8051.0 | 15541.0 | 0.0 | 0.0 | 0.504 | 0.019 | 0.0 |
| 2010 | recorded | 8732.0 | 16324.0 | 130.0 | 250.0 | 0.0 | 0.019 | 681.0 |
| 2010 | dssat_auto | 8738.0 | 16373.0 | 138.5 | 0.0 | 0.0 | 0.019 | 687.0 |
| 2011 | null | 9353.0 | 18600.0 | 0.0 | 0.0 | 0.0 | 0.023 | 0.0 |
| 2011 | recorded | 9314.0 | 18321.0 | 130.0 | 207.0 | 0.0 | 0.023 | -39.0 |
| 2011 | dssat_auto | 9340.0 | 18491.0 | 85.3 | 0.0 | 0.0 | 0.023 | -13.0 |

## Decision

- Candidate years with baseline optimization space: [2010].
- These years can enter low-cost DQN smoke testing after visual process plots are checked.
