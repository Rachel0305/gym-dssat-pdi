# 032_07 LC2010 seed0/50k stress-aware free-timing PPO cross-year transfer

## Motivation

032_06 showed that the LC2010 seed0 checkpoint at 50k timesteps is the cleanest current free-timing PPO candidate:

- It uses a compact management pattern: I45/N160.
- It avoids frequent small irrigation.
- It suppresses water/nitrogen stress reasonably well in LC2010.
- It is lower-yielding than expert in LC2010, so it must not be overclaimed.

The next low-cost question is whether this fixed policy transfers to other LC years before expanding or retuning the method.

## Scope

- Station: LCA/LC only.
- Source model: LC2010 MaskablePPO seed0 checkpoint 50k from 032_04.
- Target years: 2012, 2013, 2014, 2015.
- No retraining.
- No reward change.
- No checkpoint reselection.
- No algorithm or hyperparameter change.
- No new baseline generation unless a target year baseline is missing.

LC2011 is excluded from this run because DSSAT auto daily baseline is absent in the currently available completed-baseline files. This is a data availability exclusion, not a scientific filter.

## Fixed model

Use:

```text
benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip
```

## Fixed free-timing action/reward setting

Use the existing 032_00/032_03/032_04 stress-aware free-timing setting:

- Irrigation actions: 0, 15, 30, 45 mm
- Nitrogen actions: 0, 40, 80, 120 kg/ha
- Seasonal irrigation cap: 160 mm
- Seasonal nitrogen cap: 250 kg/ha
- Irrigation allowed DAP: 1-120
- Fertilization allowed DAP: 1-90
- Minimum interval between irrigation events: 7 days
- Minimum interval between fertilization events: 7 days

Reward used during source-model training:

```text
0.001 * [
  0.158 * delta_GRNWT
  - 1.1  * irrigation
  - 1.58 * nitrogen
  + 10.0 * irrigation * max(prev_SWFAC - current_SWFAC, 0)
  + 5.0  * nitrogen   * max(prev_NSTRES - current_NSTRES, 0)
]
```

This run only evaluates the frozen policy; it does not optimize this reward again.

## Baseline evidence

Reuse four-scenario baseline daily data:

- `031_35_full_generated_baseline_daily.csv` for null, official expert, recorded farmer.
- `031_36_full_generated_dssat_auto_daily.csv` for DSSAT auto.

Use `keep_default_na=False` when reading CSVs so that scenario label `null` is not parsed as missing.

## Outputs

For each target year:

- Frozen PPO daily CSV.
- Five-scenario combined daily CSV.
- Five-scenario endpoint summary CSV.
- Five-scenario process figure using the same style as 032_06:
  - weather,
  - soil water if available,
  - WSPD,
  - NSTD,
  - irrigation events,
  - nitrogen events,
  - grain/biomass,
  - cumulative irrigation/nitrogen.

Because 032_02 showed that older cumulative reward panels mixed incompatible reward definitions, this run must not plot mixed cumulative reward unless it recomputes one unified reward for all five scenarios. The default final panel is cumulative irrigation/nitrogen.

## Success interpretation

This is not a final model-selection experiment. It is a transfer diagnostic.

Report:

- Whether the fixed LC2010 policy produces plausible management in LC2012-2015.
- Whether yield is close to expert/recorded/auto.
- Whether water and nitrogen use are reduced.
- Whether stress becomes excessive.

Do not claim cross-year success merely from one favorable metric.

## Stop rules

- If the fixed model is missing, stop.
- If any target year lacks all four baselines, stop for that year and record the missing scenario.
- If DSSAT evaluation fails for a target year, keep the failure record and continue to other target years only if the failure is year-local.
- Do not modify model, reward, action constraints, or target years after seeing results.
