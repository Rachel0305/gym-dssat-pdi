# YC WGEN / `.CLI` Pilot Record

## Decision

**`BLOCKED_WGEN_NOT_READY`**

The training weather files pass basic format and physical checks, but no `.CLI` with verified parameters fitted only from 2004-2013 was available. A legacy YC `CNYC.CLI` exists in an older null-run folder, but its recorded window includes 2014 and its exact parameter source years are undocumented. The associated Gym-DSSAT arguments set `random_weather=false`, so that run does not prove WGEN read the file.

No new stochastic weather was generated, no WGEN seed comparison was attempted, no DSSAT WGEN season smoke was run, and no PPO training was run.

## Run Context

- Project: YC / YCA only.
- Branch at start: `codex/sya-forecast-freeze-2026-08-16`.
- HEAD at start: `ed9e7f1`.
- Existing modified files were present before this task in two smoke configs, a proposal script, and two source scripts. They were left untouched; only task outputs were staged for the task commit.
- Source FileX: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC0801.MZX`.
- Soil input: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/SOIL.SOL`.
- Historical weather source: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYCyy01.WTH`.

## Split and Year Availability

The current `055_00` config sets train to 2004-2013 and validation to 2014-2023. The YC input directory contains observed files for every year from 2000 through 2023. The additional 2000-2003 years are candidates outside the configured train and validation split, but the repository does not contain a complete audit trail of their prior model selection or manual review. A 2000-2023 YC null-run directory also exists. Therefore:

```text
independent_test_status = not_available_or_not_verified
```

The 2000-2003 years were not relabeled as test data. Machine-readable evidence is in [split_audit.json](../results/yc_wgen_cli_pilot/split_audit.json).

## YC Weather and WSTA Evidence

| Item | Evidence | Finding |
|---|---|---|
| DSSAT site / station | `CNYC0801.MZX` | YC experiment with WSTA entries such as `CNYC0801` and `CNYC1401`. |
| Historical weather station | `CNYC0401.WTH` header and year-specific files | Header station ID is `CNYC`; file names follow `CNYCyy01.WTH`. |
| Soil | YC lowIC input directory | `SOIL.SOL` is present. |
| Current Gym-DSSAT settings | `src/ppo_safe_rendering.py:336-340`; `results/yc_weather_audit/yc_weather_reset_diagnostic.json` | `random_weather=false`; auxiliary inputs are cultivar, one historical `.WTH`, and soil. No `.CLI` is passed in the current path. |
| Gym-DSSAT behavior | `references/dssat_pdi.py`; captured package version in the 001 audit | Version `0.0.5` accepts `random_weather` and a seed. It switches DSSAT weather mode to `W` when enabled and draws a new WGEN seed from that environment's RNG on reset. It does not estimate `.CLI` parameters. |

The current DSSAT WGEN source reads a climate parameter file and looks for `*CLIMA` and `*WGEN` sections. The active `055_00` FileX path uses measured weather mode, so it does not expose a runtime-verified mapping between the current 8-character year-specific WSTA (`CNYC0801`) and the WGEN parameter file. DSSAT's WeatherMan guide documents 4-character station IDs and a climate file for that station. The exact current mapping still needs confirmation in the first isolated WGEN run.

## `.CLI` Provenance

No `.CLI` exists in the current YC lowIC input directory. One plausible historical YC file is present at:

```text
DSSAT_auto_validation/run_CNYC0802_DSSAT480_IC0_null_2000_2023/pdi_smoke_test/input_used_by_pdi/CNYC.CLI
SHA256 58dbe11fdb3af9d34cc25644d8fb965627403778eb92da31a8f91619bd100d51
```

The repository contains 33 copies of this same hash under that broad run. The file contains `*WGEN PARAMETERS` and labels its source `Calculated_from_daily_data`, but its flagged-data window is 2008-01-01 through 2014-12-31 and it records 731 valid daily observations for each of RAIN, TMAX, TMIN, and SRAD. That does not establish that it was fitted only on 2004-2013. The associated PDI environment record sets `random_weather=false`. The older code refers to `my_data/CNYC.CLI`, but that referenced source file is absent in the current project.

This file is therefore excluded from the pilot. Its hash, internal date window, and exclusion reason are recorded in [yc_cli_provenance.json](../results/yc_wgen_cli_pilot/yc_cli_provenance.json).

## Train Weather Quality

The audit parsed the 10 source files for 2004-2013 and recorded each SHA256. The checks passed for all years:

- The file has the required `DATE`, `SRAD`, `TMAX`, `TMIN`, and `RAIN` fields.
- Each year has the expected 365 or 366 daily records, with continuous dates, no duplicates, and no missing dates.
- No parse errors, sentinel values, or non-finite values were found.
- Rain and solar radiation are nonnegative, and `TMAX >= TMIN` each day.
- The station ID and the year-specific filename match the YC weather source convention.
- Annual rainfall totals for all 10 years match `weather_clean/YCA_weather_cleaned.csv` within 0.1 mm.

Annual rainfall ranges from 67.4 to 802.7 mm, with a mean of 563.8 mm. 2004 is a low-rainfall review flag: 67.4 mm and a 257-day maximum dry spell. Its annual total matches the cleaned daily source exactly, so the flag indicates an unusually dry observed year rather than a demonstrated formatting error. This review flag does not fail the hard file gate, but it should be considered when reviewing WGEN parameters fitted from this 10-year period.

Files: [train_weather_qc.csv](../results/yc_wgen_cli_pilot/train_weather_qc.csv), [train_weather_source_crosscheck.csv](../results/yc_wgen_cli_pilot/train_weather_source_crosscheck.csv), and [train_weather_monthly_statistics.csv](../results/yc_wgen_cli_pilot/train_weather_monthly_statistics.csv).

## Official Route and Minimum Manual Action

DSSAT's user guide describes WGEN parameters in the station climate file and says WGEN parameters should be estimated from multiple years of daily weather. Its WeatherMan guide documents this route:

1. In WeatherMan, select the station code `CNYC` and import the 10 source files `CNYC0401.WTH` through `CNYC1301.WTH` using the IBSNAT3 daily format.
2. Select **Generate > Calculate Parameters**, set the period to 2004 day 001 through 2013 day 365, and select the WGEN parameter option.
3. Save the resulting climate file and record the WeatherMan version, exact date range, 10 source-file hashes, and output SHA256.
4. Confirm the climate filename and WSTA mapping for the current FileX before using it. Keep it in a new YC pilot input directory and leave the baseline directory untouched.

The WeatherMan steps are documented in the official [DSSAT User's Guide, Volume 3](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf). DSSAT's [User's Guide, Volume 1](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol1.pdf) describes WGEN coefficients in station `.CLI` files and the need for daily weather statistics from multiple years. The current [DSSAT WGEN source](https://github.com/DSSAT/dssat-csm-os/blob/develop/Weather/WGEN.for) parses the climate and WGEN sections from the parameter file passed to WGEN. DSSAT's [simulation code dictionary](https://github.com/DSSAT/dssat-csm-os/blob/develop/Data/SIMULATION.CDE) labels weather method `W` as internal WGEN.

No WeatherMan parameter-estimation executable or script was found inside this project. Project `AGENTS.md` limits file inspection to the project directory, so external installation directories were not inspected. No parameter file was fabricated.

## Seed Contract and Pilot Status

The future contract keeps `ppo_seed` separate from `weather_generation_seed`. The current baseline has `ppo_seed=0`; it was recorded but not used in this task. Planned weather seeds are 101-105, but none were attempted because a valid YC `.CLI` was not available. Consequently, neither same-seed identity nor different-seed divergence has been demonstrated.

| Evidence class | Status |
|---|---|
| Observed historical weather | 24 years available, 2000-2023; 10 train years audited. |
| WGEN parameter artifact (`.CLI`) | No eligible train-only artifact. The legacy candidate is excluded. |
| Stochastic weather realization | 0 generated. |
| DSSAT WGEN smoke result | Not run; there was no eligible `.CLI`. |
| PPO result | None; no PPO training was run. |

The empty pilot manifest and result tables intentionally contain headers only. See [yc_weather_manifest.csv](../results/yc_wgen_cli_pilot/yc_weather_manifest.csv), [seed_reproducibility.json](../results/yc_wgen_cli_pilot/seed_reproducibility.json), [weather_quality_summary.csv](../results/yc_wgen_cli_pilot/weather_quality_summary.csv), and [dssat_smoke_summary.csv](../results/yc_wgen_cli_pilot/dssat_smoke_summary.csv).

## Gate Summary

| Gate | Result |
|---|---|
| Train/validation split preserved | Pass: train 2004-2013, validation 2014-2023. |
| Independent test status | `not_available_or_not_verified`. |
| Historical train weather physical/format QC | Pass for 10 files; 2004 flagged for descriptive review. |
| YC `.CLI` provenance restricted to train years | Blocked. |
| 3-5 new WGEN realizations | Not run. |
| Same-seed and different-seed evidence | Not run. |
| Pilot weather statistical sanity check | Not applicable; no pilot realization exists. |
| DSSAT/Gym-DSSAT WGEN season smoke | Not run. |
| PPO training | Not run. |
| Final status | **`BLOCKED_WGEN_NOT_READY`** |

## Next Minimal Step

Use WeatherMan with only the 10 train-year files above, then preserve the generated `.CLI` and its provenance inside a new YC pilot directory. Before generating weather, verify how that `.CLI` is addressed by the current FileX WSTA field. The next automated step is a small seed test that exports or hashes the actual daily DSSAT weather sequence for one repeated seed and at least one different seed; only after those checks should the 3-5 season smokes run.

The task created only audit artifacts and this record. It did not change the original `.WTH`, soil, cultivar, FileX, `055_00`, LC/SY, HL/FQ, or existing evaluation outputs. Inherited build and audit directories were retained.
