# 015_17 HLA2010-trained DQN transfer evaluation on HLA2015

## Purpose

Evaluate whether DQN checkpoints trained on HLA2010 can produce reasonable water-nitrogen decisions on HLA2015 without additional training.

## Tested models

- HLA2010 seed0 checkpoint 35000 -> HLA2015
- HLA2010 seed1 checkpoint 25000 -> HLA2015

## Outputs

- Daily CSV: `DSSAT_auto_validation/HLA_2004/hla2010_to_2015_dqn_transfer_eval_015_17/hla2010_to_2015_transfer_eval_daily.csv`
- Summary CSV: `DSSAT_auto_validation/HLA_2004/hla2010_to_2015_dqn_transfer_eval_015_17/hla2010_to_2015_transfer_eval_summary.csv`
- Figure: `DSSAT_auto_validation/HLA_2004/hla2010_to_2015_dqn_transfer_eval_015_17/figures/hla2010_to_2015_transfer_eval_process.png`

## Results

| scenario | label | train_year | train_seed | train_checkpoint_step | final_gwad | final_cwad | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dssat_auto | DSSAT auto |  |  |  | 7648.000 | 19021.000 | 141.500 | 0.000 | 0.000 | 0.047 |  |
| expert_2007_shifted | Recorded expert |  |  |  | 7296.000 | 18639.000 | 30.000 | 165.000 | 0.779 | 0.015 |  |
| local_dqn_seed0 | Local DQN seed0 |  |  |  | 7653.187 | 19077.208 | 75.000 | 0.000 | 0.000 | 0.108 |  |
| local_dqn_seed1 | Local DQN seed1 |  |  |  | 7653.096 | 19077.202 | 75.000 | 0.000 | 0.000 | 0.076 |  |
| transfer_2010_seed0 | 2010-trained DQN seed0 | 2010.000 | 0.000 | 35000.000 | 7653.000 | 19079.000 | 90.000 | 0.000 | 0.000 | 0.096 | 1077.187 |
| transfer_2010_seed1 | 2010-trained DQN seed1 | 2010.000 | 1.000 | 25000.000 | 7653.000 | 19075.000 | 45.000 | 0.000 | 0.000 | 0.164 | 1122.187 |

## Interpretation

- If transfer models are close to local DQN, the policy has useful cross-year transfer.
- If transfer models are closer to null or use resources poorly, current DQN is mainly year-specific.