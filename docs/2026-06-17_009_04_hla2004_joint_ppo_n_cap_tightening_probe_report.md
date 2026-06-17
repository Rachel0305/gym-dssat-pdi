# 009_04 HLA 2004 Joint PPO N-Cap Tightening Probe

## Purpose

009_03 showed that lower water cost restored irrigation, but nitrogen remained near 190 kg/ha even when nitrogen cost was increased. 009_04 tests whether tightening supplemental N caps can reduce high N input while preserving yield.

## Configuration

- HLA 2004, seed 0, 5000 timesteps.
- `water_cost=0.030`, `nitrogen_cost=0.030`.
- Base N remains S1=50 kg/ha and S2=50 kg/ha.
- PPO supplemental N caps are tightened to S3=40, S4=10, S5=0 kg/ha.
- Total N upper bound is therefore 150 kg/ha.

## Comparison

| scenario | water_cost | nitrogen_cost | total_irrigation | total_n | total_base_n | total_ppo_extra_n | final_grnwt | grnwt_fraction_vs_00815_irrigation_only | swfac_days_gt_0p05 | nstres_days_gt_0p05 | soft_swfac_penalty_total | soft_nstres_penalty_total | S3_irrigation | S4_irrigation | S5_irrigation | S3_extra_n | S4_extra_n | S5_extra_n | n_cap_saturated | debug_joint_promising |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 009_02_baseline | 0.05 | 0.03 | 31.7846 | 188.9366 | 100.0 | 88.9366 | 5305.6812 | 0.7828 | 39 | 7 | 119.9664 | 4.287 | 0.0 | 15.8923 | 15.8923 | 50.0 | 19.4683 | 19.4683 | False | True |
| 009_03A_lower_water_cost | 0.03 | 0.03 | 80.0 | 190.0 | 100.0 | 90.0 | 7045.9436 | 1.0395 | 23 | 51 | 68.9905 | 38.8545 | 0.0 | 40.0 | 40.0 | 50.0 | 20.0 | 20.0 | True | False |
| 009_03B_higher_nitrogen_cost | 0.05 | 0.05 | 81.1406 | 190.0 | 100.0 | 90.0 | 7067.1625 | 1.0427 | 23 | 51 | 68.7242 | 39.8429 | 0.0 | 40.0 | 40.0 | 50.0 | 20.0 | 20.0 | True | False |
| 009_04_n_cap_tightening | 0.03 | 0.03 | 105.7213 | 150.0 | 100.0 | 50.0 | 6969.0442 | 1.0282 | 18 | 86 | 54.5631 | 70.4935 | 35.0812 | 35.3874 | 35.2528 | 40.0 | 10.0 | 0.0 | True | False |

## Interpretation

- 009_04 total N: 150.00 kg/ha.
- 009_04 total irrigation: 105.72 mm.
- 009_04 final GRNWT: 6969.04 kg/ha.
- N cap saturated: True.
- SWFAC stress days: 18; NSTRES stress days: 86.

## Decision

- 009_04 needs inspection before expanding to seed stability.

## Files

- Summary: `Leave_One_experiments/hla2004_joint_ppo_n_cap_tightening_probe_009_04/evaluation/HLA_2004_joint_ppo_summary.csv`
- Stage decisions: `Leave_One_experiments/hla2004_joint_ppo_n_cap_tightening_probe_009_04/daily_outputs/HLA/HLA_2004_seed0_soft_stress_stage_steps.csv`
- Daily CSV: `Leave_One_experiments/hla2004_joint_ppo_n_cap_tightening_probe_009_04/daily_outputs/HLA/HLA_2004_seed0_soft_stress_stage_daily.csv`
- Figure folder: `Leave_One_experiments/hla2004_joint_ppo_n_cap_tightening_probe_009_04/figures`