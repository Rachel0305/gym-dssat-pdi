# 008_16 HLA 2004 Irrigation-Only Soft-Stress PPO Seed Stability

## Purpose

Test whether the HLA 2004 irrigation-only PPO strategy found in 008_14/008_15 is reproducible across random seeds.

Do not rerun seed0. Use 008_15 seed0 as the baseline and train only seed1/seed2 at 5000 timesteps unless seed1 shows drift or degeneration.

## Strict Constraints

- Do not train all stations.
- Do not train unrestricted daily PPO.
- Do not modify `my_data/`.
- Do not modify site-packages reward files.
- Do not overwrite 008_14 or 008_15 outputs.
- Do not use hard minimum irrigation gate.
- Do not blindly increase timesteps if a seed degenerates.

## Baseline

Read seed0 from:

- `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_15_10k/evaluation/HLA_2004_stress_aware_stage_ppo_summary.csv`
- `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_15_10k/daily_outputs/HLA/HLA_2004_seed0_stress_aware_stage_steps.csv`

Expected seed0 pattern:

- S1-S3 raw irrigation action < 0;
- S4-S5 raw irrigation action > 0;
- total irrigation = 80 mm;
- no cap saturation.

## Seed1/Seed2 Settings

- Station: HLA
- Year: 2004
- Seeds: 1 and 2
- Timesteps: 5000
- Same reward/gate/action design as 008_15
- Output root:
  - `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_16_seed_stability`

## Decision Rules

After seed1:

- Stable:
  - S4/S5 raw irrigation action > 0;
  - S1-S3 raw irrigation action < 0;
  - total irrigation between 60 and 100 mm;
  - no cap saturation.
  - Then run seed2 5k.

- Drift:
  - total irrigation < 20 mm or > 110 mm;
  - or S2/S3 has positive raw irrigation and actual irrigation.
  - Do not immediately rerun. Record and analyze.

- Degeneration:
  - total irrigation = 0;
  - or cap saturation.
  - Stop and diagnose; do not continue seed2.

## Required Outputs

Save:

- seed1 and optionally seed2 models;
- daily CSV;
- stage CSV;
- training summary CSV;
- evaluation summary CSV;
- combined seed stability summary CSV;
- Markdown report:
  - `docs/2026-06-14_008_16_hla2004_irrigation_only_soft_stress_ppo_seed_stability_report.md`

## Required Interpretation

Report:

- whether seed1 is stable, drifted, or degenerated;
- whether seed2 was run;
- seed0/seed1/seed2 total irrigation;
- seed0/seed1/seed2 final GRNWT;
- raw stage action patterns;
- whether the S4/S5 late-irrigation strategy is reproducible;
- whether 008_17 second-year validation can proceed.

