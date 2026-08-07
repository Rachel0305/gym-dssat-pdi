# 053 LCA/LC lowIC site-transfer experiment prompt

## Background

SYA is currently the only station where the frozen 046_10-style PPO result looks usable. HLA, YCA, and FQA show unstable station transfer under the same framework:

- HLA had weak management contrast and required baseline-event scrutiny.
- YCA used low nitrogen and looked good mainly in PFP_N.
- FQA used high water and high nitrogen but did not gain robust yield or WP_ET superiority.

LCA/LC is opened as the final same-framework station-screening attempt.

## Objective

Run LC under the lowIC workflow:

1. `053_02`: corrected four-baseline rebuild.
2. `053_00`: frozen 046_10-style expanded-action MaskablePPO.
3. `053_01`: DSSAT native automatic irrigation plus minimal external auto-N.
4. `053_03`: yearly five-scenario daily plots and season-level bar charts.

## Controlled PPO contract

- Station code: `LCA`.
- DSSAT site/input short code: `LC`.
- Input profile: `lowIC`.
- Observation: raw 046_02 observation.
- No weather forecast features.
- No observation normalization.
- Reward/safety: inherited from the 042_15/046_02 path.
- Action grid: irrigation `[0, 15, 30, 45]` mm and nitrogen `[0, 40, 80, 120]` kg/ha.
- Train years: 2005-2013.
- Validation years: 2014-2023.
- Training: 2K smoke first, then 100K formal only after smoke gate passes.

## LC input note

`configs/sites/lca.yaml` notes that a historical single-year adapter exists for LC2010, but the current 053 station-screening series intentionally keeps the same multisite lowIC pipeline used by 050/051. Do not silently switch to the 017_11 adapter unless a separate LC-specific experiment is opened.

## Auto contract

- Irrigation: DSSAT native automatic management.
- Nitrogen: external minimal auto-N.
- Default threshold/dose: `NSTRES >= 0.5`, `25 kg/ha`.
- No DAP cutoff, no minimum interval, and no seasonal N cap.

## Reporting contract

Create yearly five-scenario daily process figures and season-level bar charts for yield, irrigation, nitrogen, WP_ET, and PFP_N.

Keep `PFP_N` as `N/A` when actual nitrogen is zero.
