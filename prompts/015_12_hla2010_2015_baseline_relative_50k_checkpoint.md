# 015_12 HLA2010/HLA2015 baseline-relative DQN 50K checkpoint prompt

## Purpose

Organize and extend the HLA2010/HLA2015 DQN line under the cross-site baseline-relative reward:

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

Each site-year uses its own null yield, but the formula, action space, budget, costs, and algorithm are shared.

## Fixed Settings

- Site-years: HLA2010 and HLA2015
- Algorithm: DQN
- Seed: 0
- Total timesteps: 50K
- Checkpoint interval: 5K
- Action space: 9 actions
- Irrigation actions: 0 / 15 / 30 mm
- Nitrogen actions: 0 / 50 / 100 kg/ha
- Budget: I <= 120 mm, N <= 300 kg/ha
- Single-event cap: I <= 30 mm, N <= 100 kg/ha
- Minimum operation interval: 7 days
- Baselines:
  - HLA2010 null = 6956 kg/ha
  - HLA2015 null = 6486 kg/ha

## Execution

Run HLA2010 first. If it completes, run HLA2015 with the same script.

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_baseline_relative_dqn_checkpoint_015_12.py --year 2010 --timesteps 50000 --seed 0 --checkpoint-interval 5000"
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_baseline_relative_dqn_checkpoint_015_12.py --year 2015 --timesteps 50000 --seed 0 --checkpoint-interval 5000"
```

## Outputs

```text
DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/
docs/2026-07-01_015_12_hla2010_baseline_relative_dqn_checkpoint_record.md
docs/2026-07-01_015_12_hla2015_baseline_relative_dqn_checkpoint_record.md
```

## Decision Rule

- If an early or middle checkpoint is better than final, report the best checkpoint.
- A successful strategy should clearly exceed null, remain close to or better than DSSAT auto/expert, and have interpretable water/nitrogen actions.
- If high yield always requires I120/N300 and reward is negative, report it as high-yield but not resource-efficient.
