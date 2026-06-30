# HLA 2004 new HY0006 cultivar forward check

## Scope

- Runtime: PDI/gym-DSSAT forward simulation.
- No PPO training.
- No parameter search.
- All scenarios keep the candidate `IC=1` setup.
- The only cultivar source used is the adjusted `MZCER048.CUL` from the calibration input package.

## HY0006 line

```text
HY0006 Haiyu    No006       . IB0001 235.2 0.494 625.3 278.0 15.50 40.00
```

## Summary

| Scenario | GWAD/HWAM proxy (kg/ha) | CWAD/CWAM proxy (kg/ha) | IRCM (mm) | NICM (kg/ha) | max WSPD | max NSTD |
|---|---:|---:|---:|---:|---:|---:|
| Candidate IC null | 0 | 3478 | 0.0 | 0.0 | 1.000 | 0.391 |
| Recorded expert replay | 269 | 5698 | 30.0 | 165.0 | 1.000 | 0.018 |
| DSSAT auto irrigation + auto-N attempt | 2814 | 8399 | 329.0 | 0.0 | 0.000 | 0.624 |

## Native DSSAT automatic-management trigger check

- Auto-irrigation events: 7; total amount = 329.00 mm.
- Auto-fertilizer events: 0; total amount = 0.00.

## Files

- `hla2004_new_cultivar_forward_check_summary.csv`
- `hla2004_new_cultivar_forward_check_daily_values.csv`
- `hla2004_new_cultivar_forward_check_management_events.csv`
- `runs/*/input/`
- `runs/*/pdi_tmp_snapshot/`

## Interpretation

The new HY0006 cultivar line was successfully propagated into each run input folder and produced changed results relative to the old cultivar line. However, the HLA 2004 low-yield behavior is not primarily caused by the cultivar line:

- Old-cultivar `candidate_recorded_expert_replay`: grain yield 242 kg/ha.
- New-cultivar `candidate_recorded_expert_replay`: grain yield 269 kg/ha.
- Old-cultivar `candidate_dssat_auto`: grain yield 2501 kg/ha.
- New-cultivar `candidate_dssat_auto`: grain yield 2814 kg/ha.

The cultivar update slightly increases grain yield, but the dominant limitation remains the candidate-IC 2004 water/nitrogen/management setup. In particular, the recorded expert replay uses only 30 mm irrigation and 165 kg/ha N under this candidate IC; water stress still reaches WSPD=1.0.

Therefore, this check supports two decisions:

1. The adjusted HY0006 cultivar file is readable by the PDI/gym-DSSAT workflow and can be used as the HLA cultivar candidate.
2. Before formal water-nitrogen optimization, the official HLA 2004 initial condition and baseline management design must be fixed. PPO training should not start from this low-yield candidate setup without first defining a realistic management/budget scenario.
