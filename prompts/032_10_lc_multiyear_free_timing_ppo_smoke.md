# 032_10 LC multi-year free-timing MaskablePPO smoke

## Objective

Run the first no-forecast LC multi-year free-timing MaskablePPO smoke test.

This task is only a workflow and feasibility smoke. It does not claim final cross-year performance.

## Scope

- Station: LCA / LC.
- Training years: 2005, 2006, 2007, 2008, 2009, 2010.
- Algorithm: MaskablePPO only.
- Weather forecast: not included.
- Training seed: 0 only.
- Total timesteps: 12,000.
- Checkpoints: 2,000, 5,000, 10,000, 12,000.
- Evaluation: deterministic evaluation of every checkpoint on each training year.

## Fixed design inherited from 032_00

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
- Reward:
  - terminal yield component: `0.158 * final_GRNWT`
  - immediate resource cost: `1.1 * irrigation + 1.58 * nitrogen`
  - stress-relief shaping:
    - `10.0 * irrigation * max(prev_SWFAC - cur_SWFAC, 0)`
    - `5.0 * nitrogen * max(prev_NSTRES - cur_NSTRES, 0)`
  - final reward scale: `0.001`

## Smoke pass criteria

Pass if:

1. Training finishes without DSSAT/container/runtime failure.
2. The random-year wrapper samples all six training years at least once.
3. All four checkpoints are saved.
4. Deterministic evaluation completes for all 4 checkpoints x 6 years.

## Interpretation boundaries

- This task does not select a final model.
- This task does not evaluate 2011-2020 or 2021-2023.
- This task does not add weather forecast information.
- This task does not change reward, action set, budget caps, or operation intervals.
- If results look bad, do not tune in-place; write a separate follow-up task.
