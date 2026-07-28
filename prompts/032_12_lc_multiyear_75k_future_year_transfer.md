# 032_12 LC multi-year 75k free-timing MaskablePPO future-year transfer

## Objective

Freeze the 032_11 LC multi-year no-forecast MaskablePPO 75k checkpoint and evaluate it on LC2011-LC2020.

This answers:

> Can the LC2005-LC2010 trained free-timing PPO candidate transfer to later LC years without retraining or weather forecast features?

## Scope

- Source model: 032_11 LC2005-LC2010 multi-year MaskablePPO seed0 checkpoint 75,000.
- Target years: LC2011-LC2020.
- Training in this task: none.
- Weather forecast: not included.
- Evaluation: deterministic frozen-policy DSSAT replay, one season per target year.
- Baseline comparison: merge against existing completed four-scenario baseline summary from 031_36 where available.

## Fixed policy design

Inherited unchanged from 032_11 / 032_00:

- Daily free-timing decisions, not expert-DAP windows.
- Action grid:
  - irrigation: 0, 15, 30, 45 mm/event
  - nitrogen: 0, 40, 80, 120 kg/ha/event
- Constraints:
  - seasonal irrigation cap: 160 mm
  - seasonal nitrogen cap: 250 kg/ha
  - irrigation allowed DAP 1-120
  - nitrogen allowed DAP 1-90
  - minimum interval between irrigation events: 7 days
  - minimum interval between fertilization events: 7 days
- Reward formula used only for evaluation logging; no learning occurs.

## Pass criteria

Pass if:

1. Source model exists and loads.
2. All target years LC2011-LC2020 have scenario-pool rows.
3. Deterministic evaluation completes for all 10 target years.
4. A baseline comparison table is written. Missing baseline cells are marked as missing rather than inferred.

## Interpretation boundaries

- This is not a re-training task.
- This is not a checkpoint-selection task.
- This does not evaluate LC2021-LC2023.
- This does not add weather forecast information.
- This does not tune reward or action constraints.
