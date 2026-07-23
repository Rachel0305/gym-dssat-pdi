# 031_35 Missing four-baseline completion for 031_34 all-year PPO transfer

## Purpose

031_34 completed frozen MaskablePPO transfer evaluation for FQA/HLA/LCA/YCA all available years after 2000, but four-scenario baseline comparison is incomplete for years that were not part of earlier screened/representative sets.

This task fills baseline coverage without changing PPO checkpoints.

## Scope

- No PPO/DQN training.
- Do not overwrite old baseline results.
- Reuse existing baseline CSVs where available.
- Generate missing fixed-policy baselines in the current all-year environment for:
  - `null`
  - `recorded_farmer_template_02705`
  - `official_extension_expert`
- Reuse true `dssat_auto` only where existing validated old forward runs are available.
- If true `dssat_auto` is not available for a station-year, mark it as missing; do not fake it with a fixed action policy.

## Boundary

`recorded_farmer_template_02705` is a transferred historical recorded-management template from the representative 027_05 year of the same site. It is not a true observed farmer record for every target year.

## Required outputs

- Unified baseline summary CSV.
- Unified baseline daily CSV for generated fixed-policy baselines.
- Coverage manifest.
- Updated PPO-vs-baseline comparison using 031_34 full PPO outputs plus the expanded baseline table.
- Experiment record MD in `docs/`.

## Smoke

Run only one missing year first: YCA2005.

## Full

If smoke passes, process all missing station-years listed by 031_34.
