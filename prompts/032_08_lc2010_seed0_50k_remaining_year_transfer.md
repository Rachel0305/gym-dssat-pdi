# 032_08 LC2010 seed0/50k PPO remaining LC-year transfer diagnostic

## Motivation

032_07 evaluated the frozen LC2010 seed0/50k free-timing stress-aware PPO model on LC2012-LC2015. The result showed strong water/nitrogen savings and higher PFP_N, but lower yields in all four transfer years. Before modifying the model, complete the remaining currently available LC years to understand the full transfer pattern.

## Scope

- Station: LCA/LC only.
- Fixed source model: LC2010 MaskablePPO seed0 checkpoint 50k from 032_04.
- Target years: 2005, 2006, 2007, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023.
- No retraining.
- No reward change.
- No checkpoint reselection.
- No algorithm or hyperparameter change.
- Reuse existing four-baseline daily data.

## Exclusions

LC2008, LC2009, and LC2011 are still excluded from this diagnostic because current completed-baseline daily files lack DSSAT auto for those years. This is a baseline availability exclusion, not a scientific or performance-based filter.

## Fixed free-timing action/reward setting

The frozen model was trained under the 032_00/032_03/032_04 stress-aware free-timing setting:

- Irrigation actions: 0, 15, 30, 45 mm
- Nitrogen actions: 0, 40, 80, 120 kg/ha
- Seasonal irrigation cap: 160 mm
- Seasonal nitrogen cap: 250 kg/ha
- Irrigation allowed DAP: 1-120
- Fertilization allowed DAP: 1-90
- Minimum interval between irrigation events: 7 days
- Minimum interval between fertilization events: 7 days

Training reward of the fixed source model:

```text
0.001 * [
  0.158 * delta_GRNWT
  - 1.1  * irrigation
  - 1.58 * nitrogen
  + 10.0 * irrigation * max(prev_SWFAC - current_SWFAC, 0)
  + 5.0  * nitrogen   * max(prev_NSTRES - current_NSTRES, 0)
]
```

This run evaluates only; it does not optimize this reward.

## Outputs

For each target year:

- Frozen PPO daily CSV.
- Five-scenario combined daily CSV.
- Five-scenario endpoint summary CSV.
- Five-scenario process PNG/SVG, using the same style as 032_07.

As established in 032_02, do not plot mixed cumulative reward. Use cumulative irrigation/nitrogen instead.

## Interpretation

This is still a transfer diagnostic, not a final model-selection result.

The result should identify:

- years where LC2010 policy transfers acceptably,
- years where yield drops too much,
- years where NSTD or WSPD becomes excessive,
- whether a multi-year LC training protocol is justified for 032_09.

## Stop rules

- If the source model is missing, stop.
- If a target year lacks all five scenarios after merging baselines and PPO, record the missing scenario and do not fabricate data.
- If one year fails DSSAT evaluation, preserve the error record and continue only if the failure is year-local.
- Do not change target years, model, reward, or constraints after seeing results.
