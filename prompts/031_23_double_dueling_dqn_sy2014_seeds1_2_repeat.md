# 031_23 Double-Dueling DQN SY2014 seed1/seed2 repeat

## Purpose

031_22 showed that mask-aware Double-Dueling DQN seed0 produced slightly higher yield than PPO and a more distributed nitrogen timing pattern, but used more irrigation and did not exceed PPO in simple profit.

This task repeats only the Double-Dueling DQN part for seed1 and seed2.

## Scope

- SYA2014 only.
- New training: Double-Dueling DQN seed1 and seed2 only.
- No PPO retraining.
- No vanilla DQN retraining.
- Same action grid, reward, caps, 7-day intervals, irrigation DAP 1-120, fertilization DAP 1-90 as 031_22.
- Final model only; no checkpoint selection.
- No reward tuning or hyperparameter tuning.

## Fixed settings

Total timesteps: 20,000 per seed.

Reward:

```text
reward = 0.001 * (0.158 * final_yield - 1.1 * irrigation - 1.58 * nitrogen)
```

## Interpretation

This task asks whether the 031_22 seed0 pattern is stable:

- high yield near or above PPO;
- not front-loading all N in DAP1-10;
- manageable N stress;
- reasonable irrigation total.

This task does not decide cross-year robustness. If seed1/2 are promising, the next step is frozen transfer to SY2012/SY2015.

