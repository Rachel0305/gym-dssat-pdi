# 031_20 Free-timing discrete MaskablePPO scaled-reward five-station smoke

## Purpose

031_17/031_18 showed that the scaled-reward discrete MaskablePPO configuration is the strongest current free-timing RL branch on SYA2014. 031_19 showed that a fair 20k DQN seed0 smoke under the same free-timing setup was not competitive with PPO.

031_20 tests whether the same fixed PPO configuration can run across all five stations on one representative year per station.

This is an engineering and behavior smoke, not a final all-year claim.

## Site-years

Use one already available year per station:

- FQA2016
- HLA2015
- LCA2010
- SYA2014
- YCA2014

These are not selected as final-success years. They are used only to check cross-station execution and gross behavior before launching all-years training.

## Fixed algorithm

- Algorithm: `sb3_contrib.MaskablePPO`
- Seed: 0
- Timesteps: 20,000 per site-year
- Daily free timing; no expert DAP windows
- Same discrete action grid, masks, caps, and reward scale as 031_17/031_18

## Reward

```text
unscaled non-terminal = -1.1 * irrigation - 1.58 * nitrogen
unscaled terminal     = 0.158 * final_grnwt - 1.1 * irrigation - 1.58 * nitrogen
training reward       = 0.001 * unscaled reward
```

The reward scale is numerical conditioning only. It does not change season-return ordering.

## Action grid and constraints

Discrete action grid:

- irrigation: `[0, 6, 12, 18, 24]` mm
- nitrogen: `[0, 40, 80, 120, 160]` kg/ha

Constraints:

- no-op always valid;
- season irrigation cap: 160 mm;
- season nitrogen cap: 250 kg/ha;
- irrigation allowed DAP 1-120;
- nitrogen allowed DAP 1-90;
- same-resource minimum interval: 7 days;
- illegal/clipping-dependent actions are masked before policy selection.

## Interpretation

Record, for every site-year:

- final grain yield;
- total irrigation;
- total nitrogen;
- simple profit;
- PFP_N;
- early DAP1-10 irrigation and nitrogen;
- stress days;
- full action sequence.

Branch rules:

- A: If most site-years run successfully and show plausible high-yield resource behavior, proceed to a broader all-year plan.
- B: If execution works but multiple stations show early-dump or poor-resource behavior, keep PPO branch but add a decision-rationality diagnostic before all-year expansion.
- C: If training/evaluation fails on multiple stations, stop and debug environment generality first.

Do not tune reward coefficients or PPO hyperparameters inside this task.

## Required outputs

- `docs/031_20_free_timing_discrete_maskableppo_scaled_reward_five_station_smoke_record.md`
- copied record under benchmark result directory
- training summary CSV
- evaluation summary CSV
- daily train/eval CSVs for each site-year
- model files for each site-year

