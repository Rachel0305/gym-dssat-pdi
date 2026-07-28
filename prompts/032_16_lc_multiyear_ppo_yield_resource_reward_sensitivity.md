# 032_16 LC multiyear PPO yield-resource reward sensitivity

## Purpose

Test whether the current LC multiyear 75k MaskablePPO result is overly resource-saving, and whether a small pre-registered reward shift can recover yield while retaining water/N savings.

## Scope

- Station: LC / LCA.
- Train years: LC2005-LC2010.
- Transfer years: LC2011-LC2020.
- Algorithm: free-timing no-forecast MaskablePPO.
- Same action grid, masks, budget limits, min operation interval, and training framework as 032_11/032_12.
- No new DSSAT baseline generation.
- No checkpoint cherry-picking outside the pre-registered 50k endpoint.

## Variants

Run exactly these variants:

1. `current_50k`
   - Reuse existing 032_11 50k checkpoint.
   - No new training.
2. `yield_plus_50k`
   - New 50k training.
   - Increase yield coefficient from `0.158` to `0.180`.
   - Keep water/nitrogen cost unchanged.
3. `resource_cheaper_50k`
   - New 50k training.
   - Keep yield coefficient `0.158`.
   - Reduce water cost from `1.1` to `0.8`.
   - Reduce nitrogen cost from `1.58` to `1.2`.

No additional variants may be added after seeing results.

## Evaluation

For each variant:

- Evaluate checkpoint 50k deterministically on LC2005-LC2010 and LC2011-LC2020.
- Compare with the four baselines:
  - null
  - recorded_farmer
  - dssat_auto
  - official_extension_expert

## Primary question

Does either yield-biased variant increase mean yield relative to the current 50k/75k resource-saving behavior without losing the main water/N-saving advantage?

## Guardrails

A variant is considered worth extending to 75k/100k only if:

- mean yield gap vs four-baseline max improves relative to `current_50k`, and
- mean irrigation remains below official expert, and
- mean nitrogen remains below official expert, and
- action sequence does not show obvious one-day all-budget dumping.

This task is a screening task only. It does not declare a final model.

## Outputs

Write under:

`benchmark_results/032_16_lc_multiyear_ppo_yield_resource_reward_sensitivity/`

Required:

- per-variant train/eval CSVs;
- combined per-year comparison CSV;
- variant summary CSV;
- record MD in `docs/`.
