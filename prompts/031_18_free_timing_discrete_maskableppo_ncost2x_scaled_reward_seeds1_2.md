# 031_18 Free-timing discrete MaskablePPO scaled-reward cross-seed repeat

## Purpose

031_17 showed that pure reward scaling (`reward_scale=0.001`) substantially improved the SYA2014 seed0 free-timing discrete MaskablePPO result. It raised yield, reduced irrigation, and avoided the failed low-yield pattern from 031_16.

031_18 tests whether that improvement is reproducible across independent seeds.

## Scope

- Site-year: SYA2014 only.
- Seeds: 1 and 2.
- Training length: 20,000 timesteps per seed.
- Algorithm: `sb3_contrib.MaskablePPO`.
- Daily free timing; no expert DAP windows.
- Same action grid, action masks, caps, PPO hyperparameters, and scaled reward as 031_17.
- No reward coefficient tuning, no action-grid change, no site/year expansion.

## Fixed configuration

Reward:

```text
unscaled non-terminal = -1.1 * irrigation - 1.58 * nitrogen
unscaled terminal     = 0.158 * final_grnwt - 1.1 * irrigation - 1.58 * nitrogen
training reward       = 0.001 * unscaled reward
```

Action grid:

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
- actions that would require clipping are masked out.

## Reference for interpreting results

031_17 seed0:

- Y=10943.10 kg/ha
- I=108 mm
- N=240 kg/ha
- simple profit=9635.10
- PFP_N=45.60
- N still front-loaded by DAP8.

Known references:

- 031_05 uniform/expert-window budget: Y=10908.60, I=160, N=250, simple profit=9498.60.
- 031_12 staged N200 counterfactual: Y=10882.18, I=72, N=200, simple profit=9810.18.

## Branch rules

- A: if both seed1 and seed2 reproduce high-yield, reduced-input behavior comparable to 031_17, this branch is promising enough for broader site-year tests.
- B: if only one of seed1/seed2 reproduces the result, mark as seed-sensitive; do not expand before a stability decision.
- C: if neither seed reproduces the result, freeze scaled-reward discrete MaskablePPO as seed0-only positive.

Do not tune parameters after seeing seed1/seed2.

## Required outputs

- `docs/031_18_free_timing_discrete_maskableppo_ncost2x_scaled_reward_seeds1_2_record.md`
- copied record under `benchmark_results/031_18_free_timing_discrete_maskableppo_ncost2x_scaled_reward_seeds1_2/`
- evaluation summary CSV
- daily train/eval CSVs for both seeds
- resolved config copy
- model files for both seeds

