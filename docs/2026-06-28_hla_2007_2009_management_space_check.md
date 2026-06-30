# HLA 2007/2009 management-response space check

## Scope

- PDI/gym-DSSAT forward simulations only.
- No PPO training.
- New HY0006 cultivar line from the corrected calibration input package.
- IC=1 is kept through the year-matched IC levels in `CNHL0701_corrected_IC123.MZX`.
- `recorded` is the original field recorded management and can temporarily serve as the expert/reference management.

## HY0006 line

```text
HY0006 Haiyu    No006       . IB0001 235.2 0.494 625.3 278.0 15.50 40.00
```

## Summary

| year | scenario | HWAM/GWAD | CWAM | IRCM | NICM | max WSPD | max NSTD |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2007 | null | 7830 | 19528 | 350.0 | 0.0 | 0.895 | 0.129 |
| 2007 | recorded | 7986 | 19809 | 363.0 | 0.0 | 0.000 | 0.014 |
| 2007 | dssat_auto | 7973 | 19638 | 363.0 | 0.0 | 0.000 | 0.032 |
| 2009 | null | 8695 | 18110 | 317.0 | 0.0 | 0.000 | 0.137 |
| 2009 | recorded | 8695 | 18040 | 316.0 | 0.0 | 0.000 | 0.019 |
| 2009 | dssat_auto | 8695 | 18110 | 317.0 | 0.0 | 0.000 | 0.137 |

## Files

- `hla_2007_2009_management_space_summary.csv`
- `hla_2007_2009_management_space_daily.csv`
- `hla_2007_2009_management_space_events.csv`
- `runs/*/input/`
- `runs/*/pdi_tmp_snapshot/`

## Corrected event-based summary

The raw `Summary.OUT` parser can be misaligned for this wide DSSAT table, so the corrected interpretation uses:

- final `GWAD/CWAD` from `PlantGro.OUT`
- irrigation and fertilizer totals from de-duplicated `MgmtEvent.OUT`

| year | scenario | GWAD | CWAD | irrigation | N | max WSPD | max NSTD |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2007 | null | 7830 | 19528 | 0.0 | 0.0 | 0.895 | 0.129 |
| 2007 | recorded | 7986 | 19809 | 30.0 | 165.0 | 0.000 | 0.014 |
| 2007 | dssat_auto | 7973 | 19638 | 93.5 | 0.0 | 0.000 | 0.032 |
| 2009 | null | 8695 | 18110 | 0.0 | 0.0 | 0.000 | 0.137 |
| 2009 | recorded | 8695 | 18040 | 20.0 | 138.0 | 0.000 | 0.019 |
| 2009 | dssat_auto | 8695 | 18110 | 47.8 | 0.0 | 0.000 | 0.137 |

## Interpretation

The 2007 and 2009 cases are useful for cultivar calibration/validation, but they show limited room for water-nitrogen optimization under the current IC=1 setup:

- In 2007, null already reaches 7830 kg/ha, while recorded management reaches 7986 kg/ha. The gain is only about 156 kg/ha.
- In 2009, null, recorded, and DSSAT auto all reach 8695 kg/ha in `GWAD`, meaning the management response is essentially flat.
- Recorded management does reduce stress indices, especially NSTD, but this does not translate into a large yield gain in these two years.

Therefore, 2007/2009 are not strong candidates for demonstrating PPO water-nitrogen optimization benefits. They are better suited as cultivar calibration/validation years. HLA 2004 remains more suitable as a stress-response optimization year, but its baseline/management-budget design must be handled carefully.

Generated diagnostic figures:

- `figures/hla_2007_management_space_rain_stress_mgmt_yield.png`
- `figures/hla_2009_management_space_rain_stress_mgmt_yield.png`
