# 231 HL/FQ validation revision and SY/LC seed expansion

## Scope and frozen sources

- HL WGEN PPO: `results/hla_wgen_8seed_053` (eight 100K models and ten validation years).
- FQ WGEN PPO: `results/fqa_multiyear_wgen_ppo_052_8seed` (eight 100K models and ten validation years).
- SY seed0: `benchmark_results/046_10_sya_originIC_expanded_action_maskableppo`.
- LC seed0: `benchmark_results/053_00_lca_lowIC_expanded_action_maskableppo`.
- No frozen source above is modified by this revision.

## HL automatic-management audit and replot

All 80 archived HL PPO validation `fileX.MZX` snapshots have `FERTI=L` in the management switch, i.e. their fertilizer operations are management events. The archived five-scenario report additionally included `dssat_auto_external_n`, a comparison baseline with automatic irrigation plus external N. It is excluded from the new four-scenario presentation; no PPO model or validation replay was altered.

- Script: `src/plot_hla_wgen_no_auto_231.py`.
- New output: `results/hla_wgen_8seed_053_no_auto_baseline_231`.
- Each seed has ten-year metric figures, yearly table, and ten-year means for Null, recorded template, official expert, and PPO.

## FQ 2018 zero-yield diagnosis and isolated revision

The frozen 2018 WTH has 15 anomalous negative summer TMIN entries (DOY 213–242). The isolated 088 correction changes only those TMIN signs; the original WTH is untouched. The correction manifest records source/corrected SHA256 and the exact rule. The original 2018 validation has zero yield in all five archived scenarios, which creates visually empty yield bars.

- Correction manifest: `benchmark_results/088_00_fq2018_input_quality_correction/088_00_correction_manifest.json`.
- Replay script: `src/replay_fqa_wgen_corrected2018_231.py`.
- Each of the eight WGEN 100K models was replayed for 2018 separately on the corrected input. DSSAT snapshots, daily data, exact Summary.OUT/ETCP metrics, model hash and weather hash are stored in `results/fqa_wgen_052_corrected2018_231/seed_XX`.
- Ten-year report script: `src/plot_fqa_wgen_corrected2018_231.py`.
- The new ten-year four-scenario report uses frozen 2014–2017 and 2019–2023 plus corrected 2018. It includes Null, recorded template, official expert and PPO. The 088 `dssat_auto` baseline has a different N contract from the frozen 052 `dssat_auto_external_n` baseline, so the auto baseline is excluded rather than mixed across years.
- Corrected 2018 PPO yields across seeds 0–7 are 7357, 7340, 7357, 7372, 8310, 7340, 8307, 8307 kg/ha. These are revised simulation outputs on a documented correction, not newly observed station values.

## SY/LC eight-seed expansion

The frozen 046_10 and 053_00 wrappers read `seed` from JSON, but the underlying 032_22 training engine uses a separate module-level `SEED=0`. The isolated launcher `src/run_sy_lc_8seed_extension.py` explicitly sets that engine seed in a fresh Python process. It creates a distinct config and output root for each seed 1–7, retaining the frozen seed0 directories.

- Both station seed1 2K smoke runs passed ten-year validation and action-transfer gates. Their inventories and model names record seed1.
- `src/orchestrate_sy_lc_8seed_extension_231.py` runs sequentially, one training process at a time, first 2K smoke then 100K formal for each seed. It checks the gate, checkpoint, actual seed and model file before advancing; any failure stops the queue.
- Progress: `results/sy_lc_8seed_extension_231/orchestrator_events.jsonl`; per-run logs are under `results/sy_lc_8seed_extension_231/logs`.
- First attempt at SY seed1 smoke used the extension JSON as the base 032_22 config and stopped with `KeyError: 'paths'` before training. That partial attempt is retained under `benchmark_results/230_sy_seed1_..._smoke2k`. The launcher was corrected to change only the engine seed; the `231` smoke then passed.

The extension is complete only when the orchestrator emits `complete` and all fourteen new 100K formal audits pass. Until then, no eight-seed SY/LC performance conclusion is made.

## 2026-10-09 seed2 continuation note

SY seed2 completed all four checkpoints and forty validation evaluations with `run_status=ok`, exact seed2 model files, and valid action transmission. Its behavior audit reports `not_all_positive_actions_at_dap1=false` and `multiple_nonzero_action_pairs=false`; the original orchestrator incorrectly treated those weak-policy outcomes as a technical training failure and stopped. The frozen seed2 output was retained. The continuation now checks technical integrity (complete year coverage, model files, action grid, action transmission, and successful validation) while preserving the behavioral audit result for analysis. The resumed queue skipped completed seeds 1 and 2 and began seed3 smoke; it does not retrain seed2.
