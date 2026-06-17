# 009_03 HLA 2004 Joint PPO Reward Balance Probe

## Purpose

This probe checks whether 009_02 used too little irrigation and too much nitrogen because water and nitrogen reward costs were imbalanced.

## Results

| scenario | water_cost | nitrogen_cost | total_irrigation | total_n | total_ppo_extra_n | final_grnwt | grnwt_fraction_vs_00815_irrigation_only | swfac_days_gt_0p05 | nstres_days_gt_0p05 | soft_swfac_penalty_total | soft_nstres_penalty_total | S4_irrigation | S5_irrigation | S3_extra_n | S4_extra_n | S5_extra_n | debug_joint_promising |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 009_02_baseline | 0.05 | 0.03 | 31.7846 | 188.9366 | 88.9366 | 5305.6812 | 0.7828 | 39 | 7 | 119.9664 | 4.287 | 15.8923 | 15.8923 | 50.0 | 19.4683 | 19.4683 | True |
| 009_03A_lower_water_cost | 0.03 | 0.03 | 80.0 | 190.0 | 90.0 | 7045.9436 | 1.0395 | 23 | 51 | 68.9905 | 38.8545 | 40.0 | 40.0 | 50.0 | 20.0 | 20.0 | False |
| 009_03B_higher_nitrogen_cost | 0.05 | 0.05 | 81.1406 | 190.0 | 90.0 | 7067.1625 | 1.0427 | 23 | 51 | 68.7242 | 39.8429 | 40.0 | 40.0 | 50.0 | 20.0 | 20.0 | False |

## Interpretation

- Lowering water cost changed irrigation from 31.78 to 80.00 mm.
- S4/S5 irrigation changed from 15.89/15.89 to 40.00/40.00 mm.
- SWFAC penalty changed from 119.97 to 68.99.
- Increasing nitrogen cost changed total N from 188.94 to 190.00 kg/ha.
- NSTRES penalty changed from 4.29 to 39.84.

## Files

- Combined summary: `Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/evaluation/009_03_reward_balance_probe_summary.csv`
- Output root: `Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03`