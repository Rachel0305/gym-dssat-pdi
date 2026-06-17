# 009_02 HLA 2004 Stage-Level Water-Nitrogen Joint PPO Smoke Test

## Goal

Run the first smoke test of the 009 stage-level water-nitrogen joint PPO framework.

This is a small diagnostic run only. Do not treat it as a final policy.

## Scope

- station: HLA
- year: 2004
- seed: 0
- total_timesteps: 5000
- stage-level PPO action
- no daily unrestricted PPO
- no hard minimum irrigation gate
- irrigation controlled by PPO under forecast/stress gate
- nitrogen uses diagnostic/agronomic base N plus PPO-controlled supplemental N

## Action Design

| Stage | DAP | Irrigation | Nitrogen |
|---|---:|---|---|
| S1 | 1-20 | blocked, cap 0 mm | fixed base N 50, PPO extra 0 |
| S2 | 21-45 | PPO 0-40 mm if gate allows | fixed base N 50, PPO extra 0 |
| S3 | 46-75 | PPO 0-40 mm if gate allows | PPO extra N 0-50 |
| S4 | 76-100 | PPO 0-40 mm if gate allows | PPO extra N 0-20 |
| S5 | 101-end | PPO 0-40 mm if gate allows | PPO extra N 0-20 |

Total N range:

- minimum = 100 kg ha-1
- maximum = 190 kg ha-1

## Reward

Use:

- growth reward
- terminal GRNWT reward
- water cost
- nitrogen cost mainly on PPO extra N
- soft SWFAC penalty
- soft NSTRES penalty

## Required Outputs

Save under:

```text
Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02
```

Required:

1. training summary CSV
2. evaluation summary CSV
3. daily CSV
4. stage decision CSV
5. process figure
6. stage action figure
7. Markdown report

## Required Summary Fields

- total_irrigation
- total_n
- total_base_n
- total_ppo_extra_n
- total_ppo_extra_irrigation
- final_grnwt
- swfac_days_gt_0p05
- nstres_days_gt_0p05
- soft_swfac_penalty_total
- soft_nstres_penalty_total
- first_irrigation_dap
- first_ppo_n_dap
- irrigation_cap_saturated
- n_cap_saturated
- debug_joint_promising

## Success Signal

- irrigation nonzero and non-saturated
- total N between 110 and 190
- total_ppo_extra_n > 0, or total_ppo_extra_n = 0 with NSTRES days < 30
- NSTRES days >= 30 is a warning unless yield remains high
- GRNWT does not collapse
- no S1 irrigation

## Failure Signal

- total irrigation = 0
- total N stays at base 100 and NSTRES days >= 30
- total N saturates 190 without yield justification
- irrigation saturates
- GRNWT collapses
- S1 irrigation occurs

## Do Not

- Do not modify `my_data/`.
- Do not modify site-packages reward files.
- Do not overwrite 008 outputs.
- Do not run multi-seed.
- Do not tune coefficients after seeing the first result.

