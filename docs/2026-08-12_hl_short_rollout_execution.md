# 2026-08-12 HL short-rollout update-geometry smoke

## Execution

- Container: `nifty_taussig`; interpreter `/opt/gym_dssat_pdi/bin/python`.
- Registered intervention package: `n_steps=60` and `batch_size=30`; all other PPO, input, years, seed, reward, safety, observation and action-grid settings matched the HL baseline.
- 2,000 steps; checkpoints 1K and 2K; isolated output `benchmark_results/067_00_hla_lowIC_short_rollout_smoke_D_short_rollout_smoke2k_short_rollout`.
- Effective loaded model confirmed `lr=3e-4`, `gamma=1`, `gae_lambda=1`, `n_steps=60`, `batch_size=30`, `n_epochs=5`, `ent_coef=0.01`, `net_arch=[64,64]`.

## Result

- Mean 2K validation yield: `6820.45 kg/ha`.
- Mean irrigation/N: `231 / 240`.
- `PFP_N=28.42`; same-definition `WP_ET` unavailable and not inferred.
- Off-grid rows: 0; safe-to-DSSAT mismatch: 0; non-zero action pairs: 6; mean post-DAP1 actions: 11.1/year.
- Cross-year action signatures: 1. This fails the preregistered mechanism gate despite correct action transmission.

The branch is stopped at 2K and does not enter 25K or 100K. The result supports “short rollout did not remove cross-year policy collapse” for this single seed, not a universal optimizer claim.

## Corrected full-metric replay context

The earlier HL replay provenance blocker was repaired by synchronizing the renderer input root with the lowIC engine root. The corrected full-metric replay gives the original HL 25K `WP_ET=1.514` and the `lr=1e-4` rescue `WP_ET=1.456`; it does not change this short-rollout branch's 2K mechanism failure.
