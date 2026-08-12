# YC seed=1 lr=1e-4 early-checkpoint stability validation

## Objective

Independently test the previously selected YC `learning_rate=1e-4` early-checkpoint candidate under seed `1`. The question is reproducibility of 5K/10K behavior, not a claim that PPO improves on the frozen YC baseline.

## One-factor contract

- Relative to the verified YC lr=1e-4 candidate, change only seed from `0` to `1`.
- Keep lowIC input, YCA/YC train years 2004--2013, validation years 2014--2023, raw observation/no forecast, reward, safety masks and 16-action grid (`I=[0,15,30,45]`, `N=[0,40,80,120]`) unchanged.
- Effective PPO kwargs: `learning_rate=1e-4`, `gamma=1`, `gae_lambda=1`, `n_steps=144`, `batch_size=144`, `n_epochs=5`, `ent_coef=0.01`, `clip_range=0.2`, `net_arch=[64,64]`.
- Use only container `nifty_taussig` and `/opt/gym_dssat_pdi/bin/python`, one serial process; prohibit 100K.

## Execution sequence and gates

1. Dry-run records all inherited factors, seed, and output roots.
2. Run 2K checkpoints 1K/2K in `benchmark_results/068_01_yca_lowIC_lr1e4_seed1_maskableppo_smoke2k`.
3. Smoke needs all ten daily files, zero request-to-safe/raw-to-safe/safe-to-DSSAT mismatch, no missing audit fields, grid compliance, at least three nonzero pairs, post-DAP1 actions in at least 8/10 years, at least two action signatures, and no collapse. Failure stops the branch.
4. If it passes, run only 25K with checkpoints 5K/10K/25K in `benchmark_results/068_01_yca_lowIC_lr1e4_seed1_maskableppo`.
5. Use repaired five-scenario replay for yield, `WP_ET`, `PFP_N`, irrigation and N. Do not infer `WP_ET` from training summaries. Stop at 25K if the endpoint collapses, fails the mechanism gate, or drops more than 5% in yield versus 10K.

## Deliverables and stop condition

Write manifest, isolated configs, loaded-model effective kwargs, checkpoint action audits, replay metric tables and execution record. A passing result remains only a candidate for a future independent seed/future-year validation; it is not frozen and never triggers 100K automatically.
