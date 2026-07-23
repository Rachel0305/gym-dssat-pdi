# 031_09 Literature-aligned DQN free-timing smoke

## Purpose

Test whether a DQN configuration aligned with the published crop-management DRL literature can improve the current free-timing SY2014 result.

This is not an exact reproduction of the paper, because the comparative paper cites a GitHub repository but does not identify one exact runnable script/checkpoint. This task is therefore a **literature-aligned reconstruction**:

- DQN-style discrete action value learning.
- 3-layer MLP Q network with 256 hidden units per layer.
- 25 discrete actions from a 5 x 5 water-nitrogen grid.
- Replay buffer, target network, epsilon-greedy exploration.
- Literature reward weights `k_yield=0.158`, `k_nitrogen=0.79`, `k_water=1.1`.

## Scope

- Site-year: SYA2014 only.
- Seed: 0 only.
- Training budget: 5,000 timesteps smoke, not full 3,000-4,000 episode literature-scale training.
- No expert DAP windows.
- Daily free timing remains enabled.
- No 7-day minimum interval in this task; minimum interval remains 1 day.
- Keep current project safety caps:
  - season irrigation <= 160 mm
  - season nitrogen <= 250 kg/ha
  - fertilization disabled after DAP90
- Daily DQN action grid:
  - irrigation: `{0, 6, 12, 18, 24}` mm
  - nitrogen: `{0, 40, 80, 120, 160}` kg/ha, clipped by remaining seasonal budget and safety rules

## Reward

Use the paper-style harvest/non-harvest reward form:

```text
if terminal:
    reward = 0.158 * final_grain_yield - 0.79 * applied_N - 1.1 * applied_water
else:
    reward = -0.79 * applied_N - 1.1 * applied_water
```

This differs from `031_04/031_08`, which used delta-grain reward or reward-v3 shaping.

## Pre-registered checks

The smoke is informative if it answers:

1. Does literature-aligned DQN still saturate I160/N250?
2. Does it reach both caps by DAP1-DAP10?
3. Does it outperform random baseline and the previous SB3-DQN/PPO free-timing smokes?
4. Does it approach the fixed-timing `uniform_spread` / `expert_window_budget` reference yield of about 10908 kg/ha?
5. Are the learned operations agronomically less suspicious than early dump or delayed late?

## Stop rule

Do not expand to all sites/years or longer training based on implementation success alone.

If the deterministic evaluation resembles random/early dump/delayed late, freeze this as a negative DQN reference and move back to the free-timing constraint/reward-design problem.

