# 032_05 LC2010 checkpoint-selection guardrail audit

## Purpose

Determine whether 032_04 already produced usable free-timing PPO checkpoints that were missed by the original "highest stress-aware reward" selection rule.

This is a pure offline audit. It does not retrain, re-evaluate DSSAT, or change any model.

## Inputs

- `benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/evaluation/032_04_checkpoint_eval_summary.csv`

## Baseline reference

Use the existing LC2010 expert yield reference from the 031 advisor-metric record:

- `expert_yield = 8739.0 kg/ha`

## Guardrail grids

Evaluate all 032_04 checkpoints under these pre-declared filters:

Yield floors:

- `95pct_expert`: yield >= 0.95 * expert_yield
- `90pct_expert`: yield >= 0.90 * expert_yield
- `85pct_expert`: yield >= 0.85 * expert_yield

Nitrogen-stress ceilings:

- `strict_nstress`: max_NSTRES <= 0.05
- `moderate_nstress`: max_NSTRES <= 0.15
- `loose_nstress`: max_NSTRES <= 0.30

For each yield-floor × NSTRES-ceiling pair:

1. Identify eligible checkpoints per seed.
2. Select the eligible checkpoint with the highest `reward_stress_aware_sum`.
3. If no checkpoint is eligible for a seed, mark `no_eligible_checkpoint`.

## Required outputs

- Full guardrail result table.
- Selected checkpoint table for each guardrail pair.
- Per-seed eligibility counts.
- Markdown record in `docs/`.

## Interpretation rule

- If a reasonable guardrail pair yields usable checkpoints in 2/3 or 3/3 seeds, the main problem is checkpoint selection mismatch rather than complete training failure.
- If only very loose guardrails work, the reward/learning setup remains unreliable.
- If no guardrail works for 2/3 seeds, then 032_04 did not generate sufficiently robust candidates and a new reward/algorithm task is required.
