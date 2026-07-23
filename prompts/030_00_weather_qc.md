# 030_00 Weather QC before free-timing RL

## Purpose

Before starting the new free-timing reinforcement-learning line, run a simple, reproducible weather-data QC pass across the current five-station input weather files.

This task is data QC only:

- no PPO/DQN training;
- no DSSAT simulation;
- no change to original `.WTH` files;
- no result-based editing.

## Input scope

Use current mainline weather-input sources only:

1. `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/<station>/*.WTH`
2. HLA long-year source:
   `DSSAT_auto_validation/HLA_2004/candidate_ic055_n025_null_2004_2023/runs/<year>/input/*.WTH`

Do not scan `benchmark_results/`, archives, copied run folders, or nested diagnostic snapshots.

Station naming note: `HL` input files correspond to the HLA/Hailun station in reports.

## QC rules

Flag daily records if any condition is true:

- `TMAX > 60` or `TMAX < -60`
- `TMIN > 50` or `TMIN < -80`
- `TMIN > TMAX`
- `RAIN < 0` or `RAIN > 500`
- `SRAD < 0` or `SRAD > 60`

## Correction rule

Original files must remain unchanged.

For physical-range violations in a single numeric weather field, create a corrected copy by linear interpolation from the nearest valid previous and next value in the same file and same variable.

Exception: DSSAT-style missing sentinels such as `-99` / `-99.0` must be flagged as missing data, not automatically interpolated in this simple QC pass. Long missing runs should be sent back to the source-data check instead of being silently filled.

For cross-field inconsistency such as `TMIN > TMAX`, flag it. Only correct if the offending variable is also outside its physical range; otherwise leave it as flagged-only.

## Outputs

Write outputs under:

`benchmark_results/030_00_weather_qc/`

Required files:

- `030_00_weather_files.csv`: all scanned files;
- `030_00_anomalies.csv`: all flagged daily records;
- `030_00_weather_qc_manifest.csv`: every automatic correction made;
- `weather_corrected/<station>/<source_group>/<filename>`: corrected `.WTH` copies only for files that need correction;
- `030_00_result.json`: compact summary for programmatic checks;
- `030_00_weather_year_usability.csv`: deduplicated station-year usability list for the next RL line;
- `030_00_weather_qc_record.md`: human-readable experiment record.

## Stop / pass rule

Pass if:

- all current mainline `.WTH` files are scanned;
- all anomalies are listed;
- original `.WTH` files are byte-for-byte unchanged;
- any corrected file is written only as a derived copy under `benchmark_results/030_00_weather_qc/weather_corrected/`.

## Usable-year rule for next RL line

For free-timing RL, use only station-years with usable weather:

- parseable `.WTH` daily table;
- at least 300 daily records for that station-year;
- no unresolved physical-range anomaly;
- no unresolved `-99` missing sentinel in `SRAD/TMAX/TMIN/RAIN`.

If multiple files cover the same station-year, choose a simple canonical source:

- HLA: prefer `hla_long`, because it covers the full HLA year series;
- other stations: prefer single-year `multisite_013` files over multi-year combined files.
