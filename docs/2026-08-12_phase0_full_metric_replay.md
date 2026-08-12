# 2026-08-12 HL/YC phase-0 full-metric replay

## Status

Blocked before producing any comparison metrics. The first read-only replay, HL original 25K, failed the existing endpoint provenance check and was stopped. No replay result is treated as valid and no full-metric batch was continued.

## Exact blocker

- Container: `nifty_taussig`
- Interpreter: `/opt/gym_dssat_pdi/bin/python`
- Candidate: `054_00_hla_lowIC_expanded_action_maskableppo`, checkpoint 25,000, validation year 2014
- Saved validation endpoint: `7309.2737 kg/ha`
- Current deterministic replay endpoint: `7127.0000 kg/ha`
- Existing replay tolerance: `1.0 kg/ha`
- Error: `PPO replay endpoint mismatch for HLA2014`

This is a renderer/input/model replay provenance issue, not a performance result. The phase-0 harness therefore did not generate a comparison table, did not infer `WP_ET`, and did not continue to the other candidates.

## Repair and completed replay

Root cause was confirmed in `src/054_hla_lowIC_site_transfer/run_054_02_hla_lowIC_five_scenario_figures.py`: the replay changed `engine.LOWIC_INPUT_ROOT` but did not synchronize `ppo_safe_rendering.MULTISITE_INPUT_ROOT`. The repair was backed up before editing and now assigns both roots to the lowIC input directory. A single HL2014 replay then matched the saved 7309 kg/ha endpoint, and the complete isolated replay finished for all six candidates.

| candidate | yield kg/ha | WP_ET | PFP_N | irrigation mm | N kg/ha |
|---|---:|---:|---:|---:|---:|
| HL original 25K | 6844.3 | 1.514 | 28.53 | 225.0 | 240.0 |
| HL lr=1e-4 25K | 6824.9 | 1.456 | 28.43 | 225.0 | 240.0 |
| YC original 25K | 8201.7 | 2.219 | 34.16 | 228.0 | 240.0 |
| YC lr=1e-4 5K | 8197.9 | 2.259 | 39.62 | 219.0 | 208.0 |
| YC lr=1e-4 10K | 8167.2 | 2.224 | 39.44 | 163.5 | 208.0 |
| YC lr=1e-4 25K | 5992.0 | 1.889 | 49.94 | 31.5 | 120.0 |

The complete replay output is isolated under `benchmark_results/066_phase0_full_metric_replay/` and the per-candidate figure/replay roots carry the `phase0_*_fixroot` label. The first failed replay directory remains preserved as evidence and is not used.

## Required repair before full metrics

1. Reconcile the checkpoint inventory/model path, effective config, lowIC input root, renderer root and environment construction for HL 25K.
2. Reproduce the saved endpoint within the registered tolerance for one year before expanding to all ten years or YC.
3. Only then run the fixed candidate list from `prompts/2026-08-12_phase0_full_metric_replay_hl_yc.md`.

## Related controlled runs completed after the blocker

- YC `ent_coef=0.02` 25K rescue completed, but failed the one-signature cross-year stability gate; see `docs/2026-08-12_yc_ent002_rescue25k.md`.
- HL short-rollout 2K completed, but also had one cross-year action signature; see `docs/2026-08-12_hl_short_rollout_execution.md`.
