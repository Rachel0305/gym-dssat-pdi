# SY seed1 continuation and LC seed2 validation - execution record

## Scope and safety

All approved tasks ran serially in `nifty_taussig` using `/opt/gym_dssat_pdi/bin/python`. New `070_*` roots were used; frozen 046_10, 053_00, 053_03, and 069 outputs were not overwritten. Syntax and dry-runs passed before training. At launch, container free memory/disk were about 14.8 GB/201 GB; SY RSS stayed below about 0.5 GB.

## SY seed1 continuation

The passed `069_00` 2K seed1 model was loaded and trained for remaining 98K nominal steps with `reset_num_timesteps=false`. Input provenance, raw/no-forecast observation, split, reward/safety, PPO settings, and the 16-action grid were unchanged from 046_10. This is a continuation, not a new independently initialized seed1 run.

All four checkpoints and 40 validation daily files were written under `benchmark_results/070_00_sya_originIC_seed1_maskableppo_100k/`. Each had 10/10 daily files/required fields, legal grid, and zero request-to-safe, raw-to-safe, and safe-to-DSSAT mismatches.

| checkpoint | nonzero pairs | post-DAP1 years | cross-year signatures | strict gate |
|---:|---:|---:|---:|---|
| 25K | 3 | 10/10 | 1 | fail |
| 50K | 4 | 10/10 | 1 | fail |
| 75K | 3 | 10/10 | 1 | fail |
| 100K | 5 | 10/10 | 2 | pass |

`WP_ET` was not inferred from the training summary.

## SY five-scenario replay - blocked

The first isolated replay (25K, SY2014) used the 070 inventory model, seed 1, and originIC input. The replay runner explicitly set `ppo_safe_rendering.MULTISITE_INPUT_ROOT` to originIC. Its endpoint was **10901.0 kg/ha**, while the saved validation endpoint was **10803.2568 kg/ha** (difference **97.7432 kg/ha**), far above the 0.5 kg/ha rounding tolerance.

The replay reproduced the saved visible management sequence (DAP1 I45/N80; DAP8 I0/N120; DAP42 I45/N0; DAP49 I45/N0), so this is not an action-grid or transmission error. It is a replay/provenance non-closure. The runner stopped immediately: no other years/checkpoints, aggregate metrics, `WP_ET`, PFP-N, wins, or performance claims were produced. The partial root `benchmark_results/070_01_sya_seed1_five_scenario_ckpt25000/` is failure evidence only. A later isolated replay must audit all environment and DSSAT staging paths; the endpoint tolerance must not be relaxed.

## LC seed1 diagnosis and seed2 smoke

Read-only LC seed1 1K/2K daily files had 10/10 complete records, zero four-way grid/transmission mismatch, 4/5 nonzero pairs, and post-DAP1 actions in every year, but one cross-year signature. Seed0 formal 50K/75K/100K similarly had one signature (only seed0 25K had two), so the signature rule is a strict robustness diagnostic rather than a renderer/action failure test. Daily files do not retain action probabilities, so reported entropy is empirical positive action-pair entropy, not statewise policy entropy.

LC seed2 changed only seed 0 to 2 and kept 053_00 lowIC/raw/no-forecast, 16-grid, split, reward/safety, and PPO settings. Its 2K final gate had 10/10 daily files/fields, zero off-grid/request-to-safe/raw-to-safe/safe-to-DSSAT mismatches, five nonzero pairs, and post-DAP1 actions in 10/10 years. It had one repeated signature (`I0/N40;I0/N80;I15/N0;I15/N120;I30/N0`), therefore the strict gate failed and `next_step_allowed=false`. No LC long training, five-scenario replay, or `WP_ET` inference ran.

## Artifacts

- Plan/runner: `configs/070_sy_lc_next_validation.json` and `src/run_070_sy_seed1_long_lc_seed2_validation.py`.
- Prompts: `prompts/2026-08-13_sy_seed1_long_and_replay.md` and `prompts/2026-08-13_lc_seed2_smoke_and_seed1_diagnostic.md`.
- Outputs: `benchmark_results/070_00_sya_originIC_seed1_maskableppo_100k/`, `benchmark_results/070_01_sya_seed1_five_scenario_ckpt25000/`, and `benchmark_results/070_02_lca_lowIC_seed2_maskableppo_smoke2k/`.
