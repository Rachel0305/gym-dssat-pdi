# 031_13 Literature-aligned DQN interval7 with doubled nitrogen reward cost

## Purpose

031_12 showed that under fixed DQN irrigation, a staged N200 schedule often keeps yield close to or above the learned early N250 schedule while improving simple profit and PFP_N.

This does **not** justify setting a site-year-specific N200 cap. Instead, 031_13 tests a uniform reward preference:

> Keep the N250 cap available, but make extra nitrogen more costly in the DQN reward.

## Frozen from 031_10/031_11

- Algorithm: literature-aligned DQN reconstruction.
- Site-year: SYA2014.
- Seeds: 0, 1, 2.
- Training budget: 5,000 timesteps per seed.
- Daily free timing; no expert DAP windows.
- Same-resource minimum operation interval: 7 days.
- Seasonal caps: I <= 160 mm, N <= 250 kg/ha.
- DAP > 90: fertilization disabled.
- Discrete action grid:
  - irrigation `{0, 6, 12, 18, 24}` mm
  - nitrogen `{0, 40, 80, 120, 160}` kg/ha
- Q network: 3 x 256 MLP.

## Single changed variable

Nitrogen cost in the literature-style reward:

```text
old: nitrogen_cost = 0.79
new: nitrogen_cost = 1.58
```

Reward becomes:

```text
if terminal:
    reward = 0.158 * final_grain_yield - 1.58 * applied_N - 1.1 * applied_water
else:
    reward = -1.58 * applied_N - 1.1 * applied_water
```

## Checks

Compare against 031_10/031_11:

1. Does total N decrease from 250?
2. Does yield remain near or above the fixed-timing reference around 10908.60 kg/ha?
3. Does simple profit improve?
4. Does PFP_N improve?
5. Does DAP1-DAP10 early heavy N weaken?
6. Does the policy avoid collapse to no-N/low-yield?

## Stop rule

Do not scan nitrogen costs. This task tests one pre-registered 2x nitrogen-cost value only.

