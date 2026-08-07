# 051 FQA/FQ originIC site-transfer experiment prompt

## Background

SYA gave one usable stage result, but HLA and YCA showed that station response can make the current PPO appear mainly strong through low nitrogen use and PFP_N. FQ/FQA is opened as the next station-screening attempt.

The workflow must keep the 050 correction: four non-PPO baselines are regenerated with static management rows using DSSAT management level `1`, so expert and recorded multi-event schedules are not lost.

## Objective

Run the FQ station under the same controlled workflow:

1. `051_02`: corrected four-baseline rebuild.
2. `051_00`: frozen 046_10-style expanded-action MaskablePPO.
3. `051_01`: DSSAT native automatic irrigation plus minimal external auto-N.
4. `051_03`: yearly five-scenario daily plots and season-level bar charts.

## Controlled PPO contract

- Station code: `FQA`.
- DSSAT site/input short code: `FQ`.
- Input profile: `originIC`.
- Observation: raw 046_02 observation.
- No weather forecast features.
- No observation normalization.
- Reward/safety: inherited from the 042_15/046_02 path.
- Action grid: irrigation `[0, 15, 30, 45]` mm and nitrogen `[0, 40, 80, 120]` kg/ha.
- Train years: 2005-2013.
- Validation years: 2014-2023.
- Training: 2K smoke first, then 100K formal only after smoke gate passes.

## Auto contract

- Irrigation: DSSAT native automatic management.
- Nitrogen: external minimal auto-N.
- Default threshold/dose: `NSTRES >= 0.5`, `25 kg/ha`.
- No DAP cutoff, no minimum interval, and no seasonal N cap.

## Reporting contract

Create yearly five-scenario daily process figures and season-level bar charts for yield, irrigation, nitrogen, WP_ET, and PFP_N.

Keep `PFP_N` as `N/A` when actual nitrogen is zero.
