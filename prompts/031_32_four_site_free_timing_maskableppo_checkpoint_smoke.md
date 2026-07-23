# 031_32 Four-site free-timing MaskablePPO checkpoint smoke

## Purpose

Start the expansion from SYA to the remaining four stations under the currently frozen free-timing discrete MaskablePPO framework.

This task is only a smoke/checkpoint-generation test before long training. It must answer:

> Can HLA/FQA/LCA/YCA each train and evaluate at least one checkpoint with the same free-timing MaskablePPO setup that was successful enough to justify SYA cross-year evaluation?

## Scope

- Target stations and train years from 031_31:
  - HLA2015
  - FQA2016
  - LCA2010
  - YCA2014
- Reference station:
  - SYA2014 is already completed in 031_26 to 031_30 and is not retrained here.
- Algorithm:
  - free-timing discrete MaskablePPO
- This smoke uses seed0 only and a short training budget.
- No parameter tuning.
- No reward changes.
- No expert-DAP windows.
- No all-year transfer claim.

## Frozen configuration

Keep the same action/reward/safety settings as the active SYA MaskablePPO line:

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

## Smoke protocol

- Run mode: `smoke`
- Seeds: `[0]`
- Maximum training steps: 512
- Checkpoint steps: `[512]`
- Evaluate the checkpoint deterministically on the same station-year.

This is not meant to judge agronomic success. It only verifies:

1. the station-year exists in the scenario pool;
2. the cleaned weather file is reachable;
3. the MaskablePPO environment can train;
4. checkpoint saving works;
5. deterministic evaluation produces daily CSV and summary rows;
6. no station-specific crash appears before full training.

## Pass/fail criteria

031_32 passes if all four target station-years:

- produce one checkpoint file;
- finish deterministic evaluation normally;
- write daily CSV and summary CSV;
- have non-empty action/reward/stress columns.

If any station fails, stop and record the exact station and traceback. Do not begin long training until the failure is resolved.

## Next step if passed

If 031_32 passes, start 031_33 as the formal four-site checkpoint-selection run:

- same frozen configuration;
- stations HLA/FQA/LCA/YCA;
- seeds `[0, 1, 2]`;
- long checkpoint schedule to be explicitly approved before launch.

