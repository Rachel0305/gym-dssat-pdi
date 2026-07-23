# 031_15 Literature DQN interval7 ncost2x 100k seed0

## Question

031_14 showed that, under the doubled nitrogen penalty (`c_N=1.58`), the already-simulated `stage_spread_N200` counterfactual outranks the original learned `N250` DQN replay in 3/3 seeds.

031_13 used only 5,000 DQN timesteps. This task tests whether the failure to learn lower-N staged behavior is simply because 5k training was too short.

## Scope

- Site-year: SYA2014 only.
- Algorithm: same literature-aligned DQN as 031_13.
- Seed: seed0 only.
- Training timesteps: 100,000.
- Single changed variable relative to 031_13: `total_timesteps 5000 -> 100000`.
- No reward redesign, no cap change, no action-space change, no new nitrogen-cost scan.

## Fixed configuration

- Reward:

```text
non-terminal: -1.1 * irrigation - 1.58 * nitrogen
terminal: 0.158 * final_grain_yield - 1.1 * irrigation - 1.58 * nitrogen
```

- Discrete actions:
  - irrigation levels: `[0, 6, 12, 18, 24]`
  - nitrogen levels: `[0, 40, 80, 120, 160]`
- Safety constraints:
  - seasonal irrigation cap: 160 mm
  - seasonal N cap: 250 kg/ha
  - same-resource minimum interval: 7 days
  - no N after DAP90
- DQN settings: inherited from 031_13, including 3x256 MLP, replay buffer, target network, epsilon-greedy exploration, and learning rate 1e-5.

## Pre-registered interpretation

- If seed0 still evaluates at `N=250` with early-heavy N, then increasing training budget alone does not fix the lower-N ranking failure.
- If seed0 moves toward staged `N200` while maintaining comparable yield, then 5k was likely too short and seed1/2 should be tested next under the same 100k configuration.
- This is a single-budget test, not a training-step scan.

