# 032_11 LC multi-year free-timing MaskablePPO training-length check

## Objective

Extend the no-forecast LC multi-year free-timing MaskablePPO run from the 032_10 12k smoke to 100k timesteps.

The question is narrow:

> Does longer training, with the same reward and constraints, reduce the early concentrated water/N application pattern observed in 032_10?

## Scope

- Station: LCA / LC.
- Training years: 2005, 2006, 2007, 2008, 2009, 2010.
- Algorithm: MaskablePPO only.
- Weather forecast: not included.
- Training seed: 0 only.
- Total timesteps: 100,000.
- Checkpoints: 25,000, 50,000, 75,000, 100,000.
- Evaluation: deterministic evaluation of every checkpoint on each training year.

## Fixed design inherited from 032_10 / 032_00

- Daily free-timing decisions, not expert-DAP windows.
- Each training episode randomly samples one year from LC2005-LC2010.
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
- Reward unchanged:
  - terminal yield component: `0.158 * final_GRNWT`
  - immediate resource cost: `1.1 * irrigation + 1.58 * nitrogen`
  - stress-relief shaping:
    - `10.0 * irrigation * max(prev_SWFAC - cur_SWFAC, 0)`
    - `5.0 * nitrogen * max(prev_NSTRES - cur_NSTRES, 0)`
  - final reward scale: `0.001`

## Pass criteria

Pass if:

1. Training finishes without DSSAT/container/runtime failure.
2. All six training years are sampled at least once.
3. All four checkpoints are saved.
4. Deterministic evaluation completes for all 4 checkpoints x 6 years.

## Interpretation boundaries

- This is not a final model-selection result.
- This does not evaluate 2011-2020 or 2021-2023.
- This does not add weather forecast information.
- This does not change reward, action set, budget caps, or operation intervals.
- If longer training still keeps early concentrated input, do not tune in-place; write a separate follow-up task.
