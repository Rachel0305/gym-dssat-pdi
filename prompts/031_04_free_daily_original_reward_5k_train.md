# 031_04 free-daily original-reward 5k training check

## Purpose

031_01/031_02 used only 512 training steps, fewer than four full SY2014 episodes. 031_03 showed that random policies also hit the I160/N250 caps, so the short-smoke cap saturation cannot be interpreted as a learned stable policy.

This task increases the training budget to 5,000 steps to test whether PPO or DQN can learn a more reasonable strategy under the same free-daily original-reward setting.

## Scope

- Site-year: SYA2014 only.
- Seed: 0 only.
- Algorithms:
  - continuous-action PPO
  - 3x3 discrete-action DQN
- Training budget: 5,000 timesteps each.
- No expert DAP windows.
- No 7-day minimum interval.
- Fertilization allowed only before or at DAP 90.
- Season caps:
  - irrigation <= 160 mm
  - nitrogen <= 250 kg/ha
- Reward:

```text
reward_t = delta_GRNWT_t - 1.0 * irrigation_t - 5.0 * nitrogen_t
```

No terminal feasibility bonus, no `/1000` reward scaling, no TOPWT term.

## DQN discretization

DQN uses the same 3x3 action grid as 031_02:

- irrigation: `{0, 20, 40}` mm/day
- nitrogen: `{0, 40, 80}` kg/ha/day

The action safety layer clips by DAP range, daily caps, and season caps.

## Success / interpretation criteria

This task does not require beating expert/auto. It asks whether longer training escapes random cap saturation.

Report:

1. final GRNWT;
2. total irrigation and nitrogen;
3. event counts and first operation DAP;
4. whether I160/N250 caps are still saturated;
5. comparison against 031_03 random/no-op baselines.

Do not expand beyond SYA2014 seed0 unless the user explicitly approves.
