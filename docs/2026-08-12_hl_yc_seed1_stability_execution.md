# HL/YC seed=1 stability validation: execution and stopping record

## Scope and provenance

- Runtime: Docker container `nifty_taussig`, interpreter `/opt/gym_dssat_pdi/bin/python`, one serial process per run.
- Both experiments were preregistered before execution. Only the random seed changed to `1`: HL retained the registered baseline PPO settings; YC retained `learning_rate=1e-4`. The lowIC input, 2004--2013 training years, 2014--2023 validation years, reward, safety, 16-action grid, raw observation and no-forecast contract did not change.
- Isolated runner: `src/run_068_hl_yc_seed1_stability_validation.py`. It contains no 100K mode and now refuses 25K if its registered 2K smoke result did not pass.
- The renderer/input-root repair from the phase-0 replay is not needed for these stopped 2K branches. `WP_ET` is absent from their training summaries and is not inferred.

## Dry-run

Both dry-runs passed with no preflight issues. They resolved the lowIC roots and station MZX files, the intended 10/10 train-validation split, the 16-action grid, seed `1`, and the effective PPO kwargs declared in the two prompts.

## HL baseline seed=1: stopped at 2K

- Output: `benchmark_results/068_00_hla_lowIC_seed1_baseline_maskableppo_smoke2k`.
- Checkpoint 1K passed the mechanism audit (six nonzero pairs and three cross-year signatures).
- Checkpoint 2K failed the preregistered endpoint gate: all ten validation years shared one action signature (`I0/N120; I45/N0; I45/N120`), so this is all-action collapse despite three nonzero pairs within that signature.
- At 2K, all ten daily files existed; off-grid, requested-to-safe, raw-to-safe and safe-to-DSSAT mismatch rows were all zero; actions after DAP1 appeared in 10/10 years.
- Training-summary endpoint means at 2K were yield `6783.69 kg/ha`, irrigation `135 mm`, N `240 kg/ha`, and `PFP_N=28.27`. These are smoke diagnostics, not five-scenario results. `WP_ET` is unavailable.
- Runtime was `106.9 s`; process RSS increase was about `147 MB`. The 25K branch, five-scenario replay and 100K were not executed.

## YC lr=1e-4 seed=1: stopped at 2K

- Output: `benchmark_results/068_01_yca_lowIC_lr1e4_seed1_maskableppo_smoke2k`.
- Both 1K and 2K failed the mechanism endpoint gate. The 2K model used five nonzero pairs, but all ten validation years had the same action signature (`I0/N40; I0/N80; I15/N0; I45/N0; I45/N40`).
- At 2K, all ten daily files and all required audit fields existed. Grid, requested-to-safe, raw-to-safe and safe-to-DSSAT checks were all zero; actions after DAP1 appeared in 10/10 years. The failure is specifically lack of cross-year action variation, not a transmission failure.
- Training-summary endpoint means at 2K were yield `8197.25 kg/ha`, irrigation `235.5 mm`, N `240 kg/ha`, and `PFP_N=34.16`. They are not enough to establish a candidate; `WP_ET` is unavailable and not inferred.
- Runtime was `107.3 s`; process RSS increase was about `145 MB`. The 25K branch, repaired five-scenario replay and 100K were not executed.

## Decision

Neither seed=1 branch is eligible for freezing, 100K, or a performance claim. The evidence supports a seed-sensitive cross-year policy-collapse concern: correct action transmission and acceptable smoke yield do not satisfy the mechanism gate. The currently frozen HL and YC benchmark candidates remain unchanged. Any later intervention should first be a new, explicitly preregistered collapse-targeting design; it must not reinterpret these smoke-summary values as `WP_ET` or as superiority evidence.
