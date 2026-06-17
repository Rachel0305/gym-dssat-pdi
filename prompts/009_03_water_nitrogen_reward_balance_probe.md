# 009_03 Water-Nitrogen Reward Balance Probe

## Goal

Diagnose whether the 009_02 HLA 2004 joint PPO result used too little irrigation and too much nitrogen because the water/nitrogen reward costs were imbalanced.

Do not expand station-years or seeds in this task.

## Baseline

009_02 baseline:

- water_cost = 0.050
- nitrogen_cost = 0.030
- total_irrigation = 31.78 mm
- total_n = 188.94 kg ha-1
- SWFAC stress days = 39
- NSTRES stress days = 7

## Probe Configurations

Run two 5k smoke tests, both HLA 2004 seed0:

1. `009_03A_lower_water_cost`
   - water_cost = 0.030
   - nitrogen_cost = 0.030

2. `009_03B_higher_nitrogen_cost`
   - water_cost = 0.050
   - nitrogen_cost = 0.050

Do not tune additional coefficients in this task.

## Required Metrics

For each run, summarize:

- total_irrigation
- total_n
- total_base_n
- total_ppo_extra_n
- final_grnwt
- GRNWT fraction vs 008_15 irrigation-only PPO
- SWFAC stress days
- NSTRES stress days
- soft_swfac_penalty_total
- soft_nstres_penalty_total
- S4 irrigation
- S5 irrigation
- S3/S4/S5 PPO extra N
- irrigation cap saturation
- nitrogen cap saturation
- debug_joint_promising

## Interpretation Rules

If lower water cost increases S4/S5 irrigation toward 40 mm and reduces SWFAC penalty:

- water cost was suppressing irrigation.

If higher nitrogen cost lowers total nitrogen without yield collapse:

- nitrogen cost was too weak in 009_02.

If neither configuration increases irrigation:

- the next diagnosis should focus on SWFAC penalty strength or gate/action design, not cost balance.

## Do Not

- Do not run multi-seed.
- Do not run FQA/SYA.
- Do not modify `my_data/`.
- Do not modify site-packages reward files.
- Do not overwrite 009_02.

