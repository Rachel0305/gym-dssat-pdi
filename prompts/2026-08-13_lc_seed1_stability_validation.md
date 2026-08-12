# LC 053_00 Seed-1 Stability Validation

## Objective

Test whether the frozen LCA lowIC 16-action MaskablePPO result
`053_00_lca_lowIC_expanded_action_maskableppo` can be reproduced with
`seed=1`. The later `053_03` nstd050 controller is a comparison scenario only
and is not part of PPO training.

## One-factor contract

- Change only the random seed: `0 -> 1`.
- Keep the lowIC input root and `LC/CNLC0801.MZX`, train years 2005--2013,
  validation years 2014--2023, raw observation, no normalization, no weather
  forecast, reward, masks and safety constraints unchanged.
- Keep the 16-action grid: irrigation `[0,15,30,45]` mm × nitrogen
  `[0,40,80,120]` kg/ha.
- Keep the frozen PPO settings: learning rate `3e-4`, gamma `1.0`, GAE lambda
  `1.0`, `n_steps=144`, `batch_size=144`, `n_epochs=5`, entropy coefficient
  `0.01`, clip range `0.2`, and network `[64,64]`.

## Runtime and isolated outputs

- Run only inside container `nifty_taussig` using
  `/opt/gym_dssat_pdi/bin/python`; never use Windows Python.
- Use one process and run stations serially.
- Write only to `benchmark_results/069_01_lca_lowIC_seed1_maskableppo_smoke2k`.
  Never overwrite seed-0 outputs.

## Gates

1. Dry-run must verify seed, input root/MZX, split years, raw/no-forecast
   contract, 16-action grid, effective PPO kwargs and a fresh output root.
2. Train 2K only, saving/evaluating checkpoints 1K and 2K.
3. The 2K checkpoint passes only if all ten validation daily files and audit
   fields exist; all grid/request-to-safe/raw-to-safe/safe-to-DSSAT mismatches
   are zero; at least three nonzero action pairs occur; actions occur after
   DAP1 in at least 8/10 years; at least two cross-year action signatures
   occur; and the policy is not an all-action collapse.
4. On any failure, stop LC immediately. On success, stop and wait for approval;
   do not start 25K or 100K automatically.

## Reporting

Save manifest, effective config, checkpoint inventory, validation CSVs,
year-level action audit and JSON result. `WP_ET` is unavailable from training
summaries and must not be inferred; obtain it only through a later fixed
five-scenario replay.
