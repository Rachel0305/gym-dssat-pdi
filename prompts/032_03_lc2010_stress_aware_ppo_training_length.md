# 032_03 LC2010 stress-aware MaskablePPO training-length audit

## Purpose

032_00 showed that the coarse-action stress-aware PPO still front-loaded water/N after 5k timesteps. 032_01 showed that LC2010 strongly penalizes late management, but a lower-water stress-triggered rule gets a higher 032 reward than the 5k PPO replay.

This task tests whether 5k training was simply too short.

## Scope

- Site-year: LCA2010 only.
- Algorithm: MaskablePPO only.
- Seed: 0 only.
- Same action grid, safety constraints, and reward as 032_00.
- Total training timesteps: 100,000.
- Checkpoints: 10k, 25k, 50k, 75k, 100k.
- Deterministic evaluation for every checkpoint.
- No checkpoint cherry-picking as final conclusion; report the whole trajectory.

## Fixed configuration

Inherit from 032_00:

- irrigation levels `{0, 15, 30, 45}`;
- nitrogen levels `{0, 40, 80, 120}`;
- season irrigation cap 160 mm;
- season N cap 250 kg/ha;
- 7-day minimum interval for both water and N;
- irrigation DAP 1-120;
- fertilization DAP 1-90;
- stress-aware reward:

```text
0.158 * final_grnwt - 1.1 * irrigation - 1.58 * nitrogen
+ 10 * irrigation * water_stress_relief
+ 5 * nitrogen * nitrogen_stress_relief
```

then scaled by 0.001.

## Decision logic

- If longer training moves PPO toward lower-water/high-reward timing without excessive event counts, 032_00 was primarily under-trained.
- If longer training remains early-frontloaded, the reward/decision structure still favors early preventive management or fails to guide delayed savings.
- If longer training becomes fragmented or lower-yield, PPO training stability is still a concern.

This task must not tune coefficients after seeing checkpoint results.

