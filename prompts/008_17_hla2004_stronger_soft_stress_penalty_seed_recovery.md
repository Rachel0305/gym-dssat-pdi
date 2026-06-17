# 008_17 HLA 2004 Stronger Soft-Stress Penalty Seed Recovery

## Purpose

Test whether stronger SWFAC soft-stress penalty can prevent the seed1 zero-irrigation degeneration observed in 008_16.

This task is a focused reward-stability diagnostic. It should not be treated as final parameter tuning.

## Strict Constraints

- Do not train all stations.
- Do not train unrestricted daily PPO.
- Do not modify `my_data/`.
- Do not modify site-packages reward files.
- Do not overwrite 008_14, 008_15, or 008_16 outputs.
- Do not use hard minimum irrigation gate.
- First run seed1 only.
- Run seed0 under the same new penalty only if seed1 recovers to a nonzero, non-saturated irrigation strategy.

## New Penalty Configuration

Compared with 008_15/008_16:

```yaml
soft_stress_reward:
  swfac_excess_cost: 3.0
  swfac_day_cost: 1.0
```

Original configuration was:

```yaml
soft_stress_reward:
  swfac_excess_cost: 1.0
  swfac_day_cost: 0.25
```

## Run Plan

### Step 1: seed1 5k

Run HLA 2004 seed1 for 5000 timesteps.

Recovery criteria:

- total irrigation > 20 mm;
- total irrigation <= 120 mm;
- no cap saturation;
- raw irrigation actions are not all -1;
- final GRNWT exceeds fixed_I60_N150 or reaches at least 0.85 of fixed_I120_N150.

### Step 2: seed0 5k if seed1 recovers

If seed1 recovers, run seed0 for 5000 timesteps under the same stronger penalty.

Do not compare new seed1 to old seed0 as the main conclusion because the reward coefficients differ.

## Required Diagnostics

For each completed seed, record:

- total irrigation;
- total N;
- final GRNWT;
- GRNWT / fixed_I120_N150;
- raw stage irrigation actions;
- stage irrigation amounts;
- growth reward sum;
- terminal reward sum;
- soft SWFAC penalty sum;
- water cost sum;
- terminal reward share;
- stress penalty share;
- whether cap saturation occurred.

## Required Outputs

Save:

- seed-level model(s);
- daily CSV(s);
- stage CSV(s);
- combined summary CSV;
- Markdown report:
  - `docs/2026-06-14_008_17_hla2004_stronger_soft_stress_penalty_seed_recovery_report.md`

## Required Interpretation

Answer:

- Did seed1 recover from zero irrigation?
- Did stronger SWFAC penalty cause cap saturation?
- Does the penalty improve seed stability without making the rule/reward overly aggressive?
- Should the next step be seed0/seed1/seed2 stability, a milder penalty scan, or second-year validation?

