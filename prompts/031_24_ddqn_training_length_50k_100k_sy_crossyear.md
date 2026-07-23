# 031_24 Double-Dueling DQN 50k vs 100k training-length sensitivity

## Purpose

031_22/031_23 showed that the mask-aware Double-Dueling DQN can produce more reasonable nitrogen timing than PPO and one very strong seed2 result at 20k steps, but seed stability is not yet clear.

This task tests whether longer training produces a more mature and transferable model.

## Scope

- Algorithm: project-local mask-aware Double-Dueling DQN.
- Training site-year: SYA2014.
- Seeds: 0, 1, 2.
- Training budgets: 50,000 and 100,000 timesteps.
- Frozen transfer evaluation years: SYA2012 and SYA2015.
- Also evaluate the training year SYA2014.
- No reward change.
- No action-space change.
- No safety-constraint change.
- Final model only; no checkpoint selection.

## Fixed reward and constraints

Reward:

```text
reward = 0.001 * (0.158 * final_yield - 1.1 * irrigation - 1.58 * nitrogen)
```

Action grid:

- irrigation levels: `[0, 6, 12, 18, 24]` mm
- nitrogen levels: `[0, 40, 80, 120, 160]` kg/ha

Safety:

- seasonal irrigation cap: 160 mm
- seasonal nitrogen cap: 250 kg/ha
- minimum irrigation interval: 7 days
- minimum fertilization interval: 7 days
- irrigation allowed DAP: 1-120
- fertilization allowed DAP: 1-90

## Evaluation

For every seed-budget model:

- deterministic SYA2014 evaluation;
- frozen deterministic transfer to SYA2012 and SYA2015;
- parse DSSAT `Summary.OUT` for ETCP, WP_ET, and PFP_N;
- compare SYA2012/SYA2015 against existing four-baseline rows from 028_05.

## Interpretation

This task asks whether increasing training length improves:

- training-year yield/resource behavior;
- nitrogen timing reasonableness;
- cross-year transfer to SYA2012/SYA2015;
- stability across seeds.

If 100k improves training year but worsens transfer, this should be interpreted as possible overfitting to SYA2014 rather than success.

