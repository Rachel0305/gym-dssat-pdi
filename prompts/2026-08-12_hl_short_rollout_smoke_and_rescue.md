# 2026-08-12 HL short-rollout update-geometry experiment

## Objective

Test whether the HL late-training drift is related to the long rollout/update geometry. This is one pre-registered intervention package, not two independent factor claims.

## Intervention contract

Change only the optimizer geometry package: `n_steps=60` and `batch_size=30`. Keep `learning_rate=3e-4`, `gamma=1`, `gae_lambda=1`, `n_epochs=5`, `ent_coef=0.01`, `clip_range=0.2`, `net_arch=[64,64]`, seed 0, lowIC input, years, raw observation, no forecast, reward, safety and 16-action grid unchanged.

## Execution

- Use `nifty_taussig` and `/opt/gym_dssat_pdi/bin/python` only.
- First run a 2K smoke with 1K/2K checkpoints, one process and isolated output.
- Proceed to 25K with 5K/10K/25K checkpoints only if the 2K mechanism gate passes.
- Never run 100K and never overwrite 054 or any earlier rescue result.

## Mechanism gate

Ten validation-year daily outputs; zero off-grid rows; zero request-to-safe, raw-to-safe and safe-to-DSSAT mismatches; at least three non-zero action pairs; post-DAP1 actions in at least 8/10 years; and at least two cross-year action signatures.

## Performance/stability gate

At 25K, yield must not be more than 3% below the candidate's 10K checkpoint; it must not show a zero-action collapse; and at least one of `WP_ET` or `PFP_N` must be non-inferior to the original HL 25K result. Missing same-definition `WP_ET` is a declared gap, not a reason to infer it.

## Deliverables

Write `docs/2026-08-12_hl_short_rollout_execution.md`, an effective-config manifest, checkpoint summaries, action audit and resource log. Do not commit or push.
