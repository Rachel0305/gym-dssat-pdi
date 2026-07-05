# 015_11 YC2014 baseline-relative DQN 50K checkpoint prompt

## Purpose

Run a checkpoint diagnostic for YC2014 under the unified baseline-relative reward:

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

The goal is to determine whether the previous YC2014 "same yield with less N" result can be reproduced under the cross-site reward definition.

## Fixed settings

- Site-year: YC2014
- Algorithm: DQN
- Seed: 0
- Total timesteps: 50K
- Checkpoint interval: 5K
- Action space: 9 actions
- Irrigation actions: 0 / 15 / 30 mm
- Nitrogen actions: 0 / 50 / 100 kg/ha
- Budget: I <= 120 mm, N <= 300 kg/ha
- Minimum operation interval: 7 days
- Baseline: YC2014 null yield from the same input package

## Execution command

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_yc2014_baseline_relative_dqn_015_10.py --timesteps 50000 --seed 0 --checkpoint-interval 5000 --checkpoint"
```

## Outputs

```text
DSSAT_auto_validation/yc2014_baseline_relative_dqn_015_10/seed0/dqn_baseline_relative_checkpoint/
docs/2026-07-01_015_10_yc2014_baseline_relative_dqn_record.md
```

Expected output files:

- `015_10_yc2014_baseline_relative_checkpoint_daily.csv`
- `015_10_yc2014_baseline_relative_checkpoint_summary.csv`
- checkpoint models under `models/`
- `figures/yc2014_baseline_relative_checkpoint_summary.png`

## Decision rule

- If any checkpoint reaches about 9418 kg/ha with N < 300, YC2014 remains a strong cross-site success case.
- If all high-yield checkpoints require N=300, YC2014 is still a high-yield case but not a stable less-N case under baseline-relative reward.
- If final checkpoint is worse than an earlier checkpoint, report best checkpoint, not final model.
