# 031_27 SY all-year frozen MaskablePPO transfer audit

## Purpose

Extend the already selected 031_26 free-timing discrete MaskablePPO checkpoints from SY2014 training to all available SY years in the existing scenario pool. This task does not train new models and does not tune reward or hyperparameters.

## Fixed source models

Use the three frozen models selected by 031_26 validation-year checkpoint selection:

- seed0: checkpoint 20k
- seed1: checkpoint 100k
- seed2: checkpoint 50k

## Scope

- Site: SYA / SY only.
- Years: all available SYA years in `Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/all_year_weather_scenario_pool.csv`, year >= 2000.
- Expected current pool: 2005-2023 inclusive, if present in the scenario pool.
- No year screening by optimizable-space labels.
- No new training.
- No new DSSAT baseline completion in this task.

## Metrics

For each frozen model x year:

- final grain yield
- final biomass
- total irrigation
- total nitrogen
- ETCP, WP_ET, PFP_N
- max SWFAC/WSPD proxy and NSTRES/NSTD proxy
- stress-day counts > 0.05
- irrigation and nitrogen event counts
- first irrigation / first nitrogen DAP
- action sequence
- daily trajectory CSV and DSSAT snapshot path

## Baseline comparison rule

Use existing four-baseline tables only when an authoritative/recomputed four-baseline envelope already exists. For SY this currently includes SY2012, SY2014, and SY2015 from the 026/028 records. For all other SY years, mark baseline comparison as `baseline_missing_not_compared` rather than inventing baselines.

## Stop / fail conditions

- If any selected 031_26 model is missing, stop.
- If scenario pool does not contain SYA years, stop.
- If a DSSAT episode does not finish, record the failure row and continue other seed-years.
- Do not reinterpret years without four-baseline envelopes as success or failure against the four scenarios.

## Output

- `benchmark_results/031_27_sy_all_year_frozen_maskableppo_transfer/evaluation/031_27_sy_all_year_candidate_summary.csv`
- `benchmark_results/031_27_sy_all_year_frozen_maskableppo_transfer/evaluation/031_27_sy_available_baseline_comparison.csv`
- `benchmark_results/031_27_sy_all_year_frozen_maskableppo_transfer/evaluation/031_27_sy_year_seed_matrix.csv`
- daily CSVs and snapshots
- `docs/031_27_sy_all_year_frozen_maskableppo_transfer_record.md`
