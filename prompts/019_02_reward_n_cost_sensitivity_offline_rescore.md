# 019_02 Reward N Cost Sensitivity Offline Rescore

## Purpose

Before launching new DQN training, use existing checkpoint summaries to test whether increasing nitrogen cost would change checkpoint selection and reduce high-N strategies.

This is a zero-DSSAT, zero-training diagnostic.

## Question

Current reward:

```text
reward = max(0, GWAD_final - local_null_GWAD)
         - 1.0 * irrigation
         - 5.0 * nitrogen
```

Test offline alternatives:

```text
N cost = 5, 10, 20
```

Water cost remains:

```text
water_cost = 1.0
```

## Inputs

Use existing checkpoint summaries only:

- HLA2010 seed0/seed1
- YC2014 seed0/seed1
- FQ2016 seed0/seed1
- SY2014 seed0/seed1
- LC2010 seed0/seed1

## Outputs

- `DSSAT_auto_validation/reward_sensitivity_019_02/019_02_all_checkpoint_rescore.csv`
- `DSSAT_auto_validation/reward_sensitivity_019_02/019_02_best_by_site_seed_ncost.csv`
- `DSSAT_auto_validation/reward_sensitivity_019_02/019_02_site_level_interpretation.csv`
- `docs/2026-07-10_019_02_reward_n_cost_sensitivity_offline_rescore.md`

## Rules

- Do not train.
- Do not call DSSAT.
- Do not overwrite old experiments.
- This only answers whether reward selection would prefer lower-N existing checkpoints; it does not prove a newly trained model would learn them.
