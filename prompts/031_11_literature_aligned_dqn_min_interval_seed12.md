# 031_11 Literature-aligned DQN 7-day interval seed stability check

## Purpose

031_10 produced the first strong positive free-timing DQN smoke on SY2014 seed0. This task checks whether that result is seed-specific.

## Frozen configuration

Exactly reuse 031_10 except for random seed:

- Algorithm: literature-aligned DQN reconstruction.
- Site-year: SYA2014.
- Training budget: 5,000 timesteps.
- Decision timing: daily free timing, no expert DAP windows.
- Same-resource minimum operation interval: 7 days.
- Seasonal caps: I <= 160 mm, N <= 250 kg/ha.
- DAP > 90: fertilization disabled.
- Discrete action grid:
  - irrigation `{0, 6, 12, 18, 24}` mm
  - nitrogen `{0, 40, 80, 120, 160}` kg/ha
- Q network: 3 x 256 MLP.
- Literature reward:

```text
if terminal:
    reward = 0.158 * final_grain_yield - 0.79 * applied_N - 1.1 * applied_water
else:
    reward = -0.79 * applied_N - 1.1 * applied_water
```

## New runs

- seed1
- seed2

Do not change reward, network, action grid, interval, or training budget.

## Checks

Compare seed1/seed2 against seed0 031_10:

1. final grain yield;
2. total irrigation and nitrogen;
3. simple profit `Y - I - 5N`;
4. whether both caps are reached by DAP10;
5. nonzero action sequence;
6. whether repeated clipped high-N requests persist.

## Stop rule

If seed1/seed2 fail badly, do not tune in place. Record as seed instability and decide separately whether to adjust action grid/reward/interval.

