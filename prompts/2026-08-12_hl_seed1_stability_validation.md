# HL seed=1 baseline stability validation

## Objective

Test whether the current HL reference PPO behavior is reproducible under one independent random seed. This is a stability diagnostic, not a new optimizer search and not a basis for freezing a new paper result.

## One-factor contract

- Change only training/evaluation seed from `0` to `1`.
- Keep lowIC input, HLA train years 2004--2013, validation years 2014--2023, raw observation, no forecast, reward, safety masks, 16-action grid (`I=[0,15,30,45]`, `N=[0,40,80,120]`) unchanged.
- Effective PPO kwargs: `learning_rate=3e-4`, `gamma=1`, `gae_lambda=1`, `n_steps=144`, `batch_size=144`, `n_epochs=5`, `ent_coef=0.01`, `clip_range=0.2`, `net_arch=[64,64]`.
- Container/runtime only: `nifty_taussig`, `/opt/gym_dssat_pdi/bin/python`; one process and serial execution. Do not use Windows Python. Never run 100K.

## Execution sequence and gates

1. Dry-run must record station, input root, years, grid, seed, effective PPO kwargs and isolated output root.
2. Run 2K smoke with checkpoints 1K/2K in `benchmark_results/068_00_hla_lowIC_seed1_baseline_maskableppo_smoke2k`.
3. At the 2K endpoint, all 10 validation-year daily files are required. Requested-to-safe, raw-to-safe and safe-to-DSSAT mismatch counts must each be zero; missing fields are a data gap and fail the gate. Actions must stay on-grid, have at least three nonzero pairs, occur after DAP1 in at least 8/10 validation years, and exhibit at least two cross-year action signatures without all-action collapse.
4. Only if the smoke gate passes, run 25K with 5K/10K/25K checkpoints in `benchmark_results/068_00_hla_lowIC_seed1_baseline_maskableppo`.
5. Replay each 25K checkpoint candidate through the repaired five-scenario renderer before reporting `WP_ET`; training summaries are not a `WP_ET` source. Stop if the 25K endpoint fails the mechanism gate, collapses, or is more than 5% below its 10K yield.

## Deliverables and stop condition

Save configs, effective YAML, manifest, action-audit CSV/JSON and execution record. Report completed versus not executed, metric provenance, and whether the branch only merits another seed/future-year test. Do not freeze or promote a candidate and do not push GitHub.
