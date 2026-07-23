# 031_33 Four-site free-timing MaskablePPO checkpoint selection

## Purpose

Run the first formal four-site expansion of the frozen free-timing discrete MaskablePPO framework after the successful 031_32 smoke.

This task asks:

> For each target station, can the same free-timing MaskablePPO configuration produce candidate checkpoints on that station's own training year?

## Scope

- Target stations and training years:
  - HLA2015
  - FQA2016
  - LCA2010
  - YCA2014
- Reference station:
  - SYA2014 is already completed in 031_26 to 031_30 and is not retrained here.
- Seeds:
  - 0, 1, 2
- Maximum training steps per station-seed:
  - 100000
- Checkpoints:
  - 10000
  - 20000
  - 30000
  - 50000
  - 75000
  - 100000

## Frozen method

Do not tune or change the method in this task.

Reward:

```text
r = 0.001 * (0.158 * final_yield_at_harvest - 1.1 * irrigation - 1.58 * nitrogen)
```

Discrete action grid:

- irrigation: `[0, 6, 12, 18, 24]` mm
- nitrogen: `[0, 40, 80, 120, 160]` kg/ha

Safety/action constraints:

- seasonal irrigation cap: 160 mm
- seasonal nitrogen cap: 250 kg/ha
- minimum interval between irrigation events: 7 days
- minimum interval between fertilization events: 7 days
- irrigation allowed DAP: 1 to 120
- fertilization allowed DAP: 1 to 90

PPO hyperparameters:

- learning rate: 3e-4
- gamma: 1.0
- GAE lambda: 1.0
- n_steps: 144
- batch size: 144
- n_epochs: 5
- entropy coefficient: 0.01
- clip range: 0.2
- net arch: [64, 64]

## Checkpoint-selection rule

This task does not yet perform all-year transfer and does not use any held-out year for selection.

For each station-seed:

1. Evaluate all checkpoints deterministically on the same training station-year.
2. Select the checkpoint with the highest `literature_reward_sum`.
3. Tie-breakers:
   1. higher final grain yield;
   2. lower total nitrogen;
   3. lower total irrigation;
   4. earlier checkpoint step.

This is a training-year candidate selection rule. It must not be reported as cross-year generalization.

## Pass/fail criteria

031_33 passes as an execution task if:

- all four target stations finish all three seeds;
- every seed has all six checkpoint files;
- every checkpoint can be evaluated deterministically on the training year;
- exactly one selected checkpoint is written per station-seed.

Scientific interpretation is deferred to 031_34 all-year transfer.

## Stop rules

- If a station-seed fails, record the exact traceback and continue remaining station-seeds only if memory/runtime state is normal.
- Do not change reward, hyperparameters, constraints, or train years inside this task.
- Do not add a new station/year inside this task.
- Do not delete partial outputs; this script must support resume.

## Next step if passed

031_34 will freeze the selected checkpoint for each station-seed and evaluate it on all weather-available years of that station, comparing against the station-specific baseline envelope.

