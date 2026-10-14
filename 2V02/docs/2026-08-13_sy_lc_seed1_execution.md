# SY/LC frozen PPO seed=1 stability validation — 2026-08-13

## Scope and decision rule

This run tested seed stability only.  The frozen PPO protocols were retained:

- SY: `046_10_sya_originIC_expanded_action_maskableppo`, originIC;
- LC: `053_00_lca_lowIC_expanded_action_maskableppo`, lowIC.  The later
  `053_03` nstd050 external-N scenario is not PPO training.

For both sites, the sole planned factor change was seed `0 -> 1`.  Inputs/MZX,
2005--2013 train and 2014--2023 validation years, raw/no-forecast observation,
reward/safety, PPO kwargs, and the 16-action grid I `[0,15,30,45]` × N
`[0,40,80,120]` were held fixed. Container execution was serial and single
process using `/opt/gym_dssat_pdi/bin/python` in `nifty_taussig`.

The 2K gate was evaluated at checkpoint 2K: 10/10 daily files and audit fields,
zero grid/request-to-safe/raw-to-safe/safe-to-DSSAT mismatches, >=3 nonzero
action pairs, post-DAP1 actions in >=8 validation years, >=2 cross-year action
signatures, and no all-signature collapse. A passing smoke is not a claim of
cross-seed performance stability and does not authorize long training.

## Configuration provenance

Dry-run passed for both stations before training. Each loaded frozen seed-0
100K model reported the fixed MaskablePPO parameters: learning rate `3e-4`,
gamma `1.0`, GAE lambda `1.0`, `n_steps=144`, `batch_size=144`, `n_epochs=5`,
entropy coefficient `0.01`, clip range `0.2`, policy/value networks `[64,64]`.

- SY source MZX: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX`.
- LC source MZX: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC0801.MZX`.

## Completed 2K smoke results

| Site | Isolated output | Runtime | 2K nonzero pairs | Post-DAP1 years | Cross-year signatures | Transmission/grid mismatch | Gate | Long task |
|---|---|---:|---:|---:|---:|---:|---|---|
| SY | `069_00_sya_originIC_seed1_maskableppo_smoke2k_rerun1` | 94.9 s | 7 | 10/10 | 4 | 0/0/0/0 | passed | not executed |
| LC | `069_01_lca_lowIC_seed1_maskableppo_smoke2k_rerun1` | 80.4 s | 5 | 10/10 | 1 | 0/0/0/0 | failed | not executed |

SY's 1K checkpoint was collapsed (one action pair/signature and no post-DAP1
actions), but its pre-registered final 2K gate passed. Its 2K audit includes
seven legal pairs and four year-specific signatures. This verifies that seed=1
can leave the immediate collapsed state by 2K; it does **not** yet reproduce
the 100K seed-0 policy or its outcome metrics.

LC's 2K checkpoint had legal and transmitted actions (five pairs, all ten years
with post-DAP1 actions) but exactly the same five-pair signature in every
validation year. Thus `crossyear_action_signature_count=1` and
`all_action_collapse=true`; the mechanism gate failed. No LC 25K/100K training
was started.

## Execution note and artifacts

The first SY attempt created `069_00_sya_originIC_seed1_maskableppo_smoke2k`
but stopped before PPO learning because the framework attempted to copy its
effective YAML onto itself (`SameFileError`). It contains no completed model or
result and was preserved. The runner was corrected to source the effective YAML
from `configs/` outside the output `configs/` folder, and both actual smoke
runs were written to fresh `*_rerun1` roots.

- Prompt/config/runner: `prompts/2026-08-13_*_seed1_stability_validation.md`,
  `configs/069_sy_lc_seed1_stability_validation.json`,
  `src/run_069_sy_lc_seed1_stability_validation.py`.
- Per-site manifests, JSON results, checkpoint inventory, validation CSVs and
  year-level action audits are under the two `*_rerun1` output roots above.
- `WP_ET` is unavailable from these training summaries and was not inferred.
  No five-scenario replay was run.

## Current interpretation

The smoke evidence is insufficient to call either station cross-seed stable.
SY is eligible for a separately approved seed=1 long-run/checkpoint validation;
LC is not eligible for long training under the registered mechanism gate.
Neither result changes the frozen seed-0 paper result in this run.
