# 032_04 LC2010 free-timing stress-aware MaskablePPO multiseed 200k

## Purpose

Replicate the 032_03 positive signal under a fixed multi-seed protocol before expanding to other years or stations.

032_03 showed that LCA2010 seed0 can reach a lower-input candidate at 50k (`I45/N160`, yield `8337.90`, stress-aware reward `1.163588`), but later checkpoints degraded. This task tests whether the same free-timing PPO configuration is seed-stable, rather than treating seed0 as sufficient evidence.

## Fixed scope

- Site-year: LCA2010 only.
- Seeds: 0, 1, 2.
- Algorithm: MaskablePPO only.
- Training timesteps per seed: 200,000.
- Checkpoints: 10k, 20k, 25k, 50k, 75k, 100k, 150k, 200k.
- No cross-year transfer.
- No cross-station transfer.
- No DQN in this task.
- No reward/constraint/action tuning in this task.

## Decision/action constraints

Use the same free-timing stress-aware setup as 032_03:

- No expert DAP windows.
- Irrigation actions: 0, 15, 30, 45 mm.
- Nitrogen actions: 0, 40, 80, 120 kg/ha.
- Seasonal irrigation cap: 160 mm.
- Seasonal nitrogen cap: 250 kg/ha.
- Irrigation allowed: DAP 1-120.
- Nitrogen allowed: DAP 1-90.
- Minimum operation interval: 7 days.
- Over-budget and invalid actions are masked, not clipped.

## Reward

Use the unchanged 032_03 stress-aware reward:

```text
reward = 0.001 * [
    0.158 * delta_GRNWT
    - 1.1 * irrigation
    - 1.58 * nitrogen
    + stress_relief_bonus
]
```

where:

```text
stress_relief_bonus =
    10 * irrigation * max(previous_SWFAC - current_SWFAC, 0)
  + 5 * nitrogen * max(previous_NSTRES - current_NSTRES, 0)
```

This stress-relief term is project-local exploratory reward shaping. It is not a literature-standard fixed formula.

## Checkpoint selection and reporting

For each seed:

1. Report all checkpoint evaluations.
2. Mark the checkpoint with the highest `reward_stress_aware_sum` as the pre-specified selected checkpoint.
3. Also report the final 200k checkpoint separately.
4. Do not select by visual appearance, yield alone, or resource use alone.

## Required outputs

- Prompt copy under output configs.
- Resolved env config.
- Checkpoint inventory with model SHA256.
- Checkpoint evaluation CSV.
- Daily output CSV for every evaluated checkpoint.
- Markdown record in `docs/`.
- Output copy of the Markdown record under `benchmark_results/032_04.../`.

For each checkpoint, record:

- final yield
- total irrigation
- total nitrogen
- PFP_N
- stress-aware reward sum
- max SWFAC and NSTRES
- stress days at thresholds `>0.001`, `>0.01`, `>0.05`
- irrigation/nitrogen event counts
- first irrigation/N DAP
- full action sequence

## Stop/interpretation rules

- If 2/3 or 3/3 seeds select low-input, stress-controlled strategies with yield close to expert, 032_03 is likely not a seed0 accident.
- If seed outcomes diverge strongly, do not expand to all years; next task should address seed stability or checkpoint validation.
- If 150k/200k recover after 100k degradation, a long-training checkpoint-selection protocol may be justified.
- If 100k-200k continues to degrade, do not run 500k before designing early stopping/validation.
