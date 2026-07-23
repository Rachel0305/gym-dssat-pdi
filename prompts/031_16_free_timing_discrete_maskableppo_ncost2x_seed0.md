# 031_16 Free-timing discrete MaskablePPO ncost2x seed0

## Question

031_13/031_15 showed that literature-style DQN still learns early N250 even when:

- nitrogen cost is doubled to 1.58;
- training is extended from 5k to 100k for seed0.

031_14 showed that the true DSSAT counterfactual ranking under `c_N=1.58` already favors staged N200 over the learned early N250 replay.

This task tests whether a different RL algorithmic form can recover the better ranking:

> Replace DQN with free-timing discrete MaskablePPO, while keeping reward, action grid, caps, and no-expert-DAP freedom fixed.

## Scope

- Site-year: SYA2014.
- Seed: 0 only.
- Training timesteps: 20,000.
- No expert DAP windows.
- No reward coefficient scan.
- No cap tuning.
- No new deterministic counterfactual search.

## Fixed reward

Same literature-style reward as 031_13/031_15:

```text
non-terminal reward = -1.1 * irrigation - 1.58 * nitrogen
terminal reward = 0.158 * final_grain_yield - 1.1 * irrigation - 1.58 * nitrogen
```

## Action space

Discrete 25-action grid:

- irrigation levels: `[0, 6, 12, 18, 24]` mm
- nitrogen levels: `[0, 40, 80, 120, 160]` kg/ha

## Action mask and constraints

At each daily decision step, MaskablePPO may only choose actions that are legal under:

- seasonal irrigation cap: 160 mm
- seasonal nitrogen cap: 250 kg/ha
- same-resource minimum interval: 7 days
- irrigation allowed DAP: 1-120
- nitrogen allowed DAP: 1-90
- no action may rely on wrapper clipping; if a requested amount would exceed the remaining cap or violate timing constraints, it is masked out.
- no-op `(I=0,N=0)` must always remain valid.

## Algorithm

- `sb3_contrib.MaskablePPO`
- `gamma=1.0`
- `gae_lambda=1.0`
- on-policy objective, not Q-value regression.

## Pre-registered interpretation

- If MaskablePPO moves toward staged/lower-N behavior while preserving yield, DQN failure is likely algorithm-specific rather than an unavoidable reward issue.
- If MaskablePPO still converges to early N250/full caps, then the current reward/action design is insufficient even for policy-gradient training.
- This is a single algorithmic smoke. Do not expand to all sites/years unless seed0 beats the 031_13/031_15 N250 behavior and random/early-dump references.

