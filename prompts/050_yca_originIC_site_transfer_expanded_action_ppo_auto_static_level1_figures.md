# 050 YCA/YC originIC site-transfer experiment prompt

## Background

HL/HLA showed a serious comparison-risk: static expert/recorded management could appear as only one executed event, causing five scenarios to have nearly identical yield and WP_ET while PPO looked good mainly through PFP_N. Before trying another station, the non-PPO baselines must be regenerated with the static level-1 management-row correction from 037_07.

## Objective

Create a YC station series that reuses the current SYA PPO method and the HLA-style reusable workflow, but fixes the baseline-management provenance before plotting:

1. `050_02`: regenerate four non-PPO baselines for `YCA` / `YC` with static management rows rendered as management level `1`.
2. `050_00`: train MaskablePPO using the frozen 046_10 method.
3. `050_01`: evaluate DSSAT native automatic irrigation plus minimal external auto-N.
4. `050_03`: draw yearly five-scenario daily plots and season-level bar charts from the corrected baseline, auto, and PPO outputs.

## Controlled PPO contract

- Station code: `YCA`.
- DSSAT site/input short code: `YC`.
- Input profile: `originIC`.
- Observation: raw 046_02 observation.
- No weather forecast features.
- No observation normalization.
- Reward/safety: inherited from the 042_15/046_02 path.
- Action grid: irrigation `[0, 15, 30, 45]` mm and nitrogen `[0, 40, 80, 120]` kg/ha.
- Training: 2K smoke first, then 100K formal only after smoke gate passes.
- Checkpoints: 25K, 50K, 75K, 100K.

## Auto contract

- Irrigation: DSSAT native automatic management.
- Nitrogen: external minimal auto-N because DSSAT automatic fertilization is not operational in this workflow.
- Default threshold/dose: `NSTRES >= 0.5`, `25 kg/ha`.
- No DAP cutoff, no minimum interval, and no seasonal N cap.
- Alternative thresholds must be run as separate config/run-id outputs, not by overwriting the default nstd050 result.

## Baseline contract

Use `run_multisite_input_ic1_four_baseline_rebuild_034_00.py` scenario definitions, but patch `replace_static_application_rows` with `run_static_level1_four_baseline_rebuild_037_07.replace_static_application_rows_level1`.

This matters because in DSSAT `@I IDATE` and `@F FDATE` first numeric column is the selected management level ID, not an event sequence ID. Static recorded/expert schedules for one selected scenario therefore need all rows to use level `1`.

## Reporting contract

Create yearly five-scenario daily process figures and season-level bar charts for:

- grain yield;
- total irrigation;
- total nitrogen;
- WP_ET;
- PFP_N.

Keep `PFP_N` as `N/A` when actual nitrogen is zero.

Do not use old `034_00` baseline outputs for final 050 five-scenario figures unless explicitly requested for diagnosis.
