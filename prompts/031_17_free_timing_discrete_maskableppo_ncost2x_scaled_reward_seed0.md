# 031_17 Free-timing discrete MaskablePPO with scaled reward, SYA2014 seed0

## Purpose

031_16 showed a mixed result: discrete MaskablePPO with strict action masks reduced nitrogen use from N250 to N200, but still front-loaded all N by DAP9 and produced low yield. This suggests the action representation/mask helped resource restraint but did not solve free-timing credit assignment.

031_17 tests one narrow training-stability hypothesis:

> PPO value/policy learning may be impaired by the raw reward magnitude. Scaling the reward by 0.001 may improve numerical learning without changing the management objective or the ordering of season returns.

## Scope

- Site-year: SYA2014 only.
- Seed: 0 only.
- Training length: 20,000 timesteps.
- Algorithm: `sb3_contrib.MaskablePPO`.
- Daily free timing; no expert DAP windows.
- Same discrete action grid, action masks, caps, and PPO hyperparameters as 031_16.
- Single variable changed from 031_16: multiply the reward returned to PPO by `0.001`.

## Reward

Unscaled objective:

```text
non-terminal reward = -1.1 * irrigation - 1.58 * nitrogen
terminal reward     = 0.158 * final_grnwt - 1.1 * irrigation - 1.58 * nitrogen
```

Training reward:

```text
ppo_reward = 0.001 * unscaled_reward
```

Both unscaled and scaled reward values must be logged. The scale must not be interpreted as a new management preference; it is only a numerical conditioning test.

## Action grid and constraints

Discrete action grid:

- irrigation: `[0, 6, 12, 18, 24]` mm
- nitrogen: `[0, 40, 80, 120, 160]` kg/ha

Masks/constraints:

- no expert DAP windows;
- no-op always valid;
- season irrigation cap: 160 mm;
- season nitrogen cap: 250 kg/ha;
- irrigation allowed DAP 1-120;
- nitrogen allowed DAP 1-90;
- same-resource minimum interval: 7 days;
- actions that would require wrapper clipping are masked out before policy selection.

## Success interpretation

This is not a full success/failure test for the thesis. It is a single-seed diagnostic.

Compare against 031_16 and known references:

- 031_16 discrete MaskablePPO: Y=10132.14, I=156, N=200, N all by DAP9.
- 031_12 staged N200 counterfactual: Y=10882.18, I=72, N=200.
- 031_05 uniform/expert-window budget reference: Y=10908.60, I=160, N=250.

Branch rules:

- A: If scaled reward materially improves timing/yield while preserving N economy, repeat the exact configuration across seeds before any expansion.
- B: If reward/loss magnitudes improve but timing remains early/front-loaded or yield remains far below staged references, reward magnitude alone is not the main blocker.
- C: If behavior worsens or training fails, stop this scaled-reward branch.

Do not expand to all stations/years from this single seed.

## Required outputs

- `docs/031_17_free_timing_discrete_maskableppo_ncost2x_scaled_reward_seed0_record.md`
- copied record under `benchmark_results/031_17_free_timing_discrete_maskableppo_ncost2x_scaled_reward_seed0/`
- evaluation summary CSV
- daily train/eval CSVs
- resolved config copy
- model file

