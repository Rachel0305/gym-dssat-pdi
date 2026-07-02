# 015_19 HLA2010-trained DQN transfer evaluation on HLA2016 and HLA2022

## Purpose

Evaluate whether HLA2010 DQN checkpoints transfer to HLA2016 and HLA2022 without additional training.

## Outputs

- Daily CSV: `DSSAT_auto_validation/HLA_2004/hla2010_to_2016_2022_dqn_transfer_eval_015_19/hla2010_to_2016_2022_transfer_eval_daily.csv`
- Summary CSV: `DSSAT_auto_validation/HLA_2004/hla2010_to_2016_2022_dqn_transfer_eval_015_19/hla2010_to_2016_2022_transfer_eval_summary.csv`
- Figures: `DSSAT_auto_validation/HLA_2004/hla2010_to_2016_2022_dqn_transfer_eval_015_19/figures`

## Results

| requested_year | scenario | label | train_seed | train_checkpoint_step | final_gwad | final_cwad | rain_total | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016 | dssat_auto | DSSAT auto |  |  | 7538.000 | 17018.000 | 365.600 | 140.800 | 0.000 | 0.000 | 0.022 |  |
| 2022 | dssat_auto | DSSAT auto |  |  | 7935.000 | 18552.000 | 360.400 | 143.600 | 0.000 | 0.000 | 0.012 |  |
| 2016 | null | Null |  |  | 7224.000 | 16612.000 | 365.600 | 0.000 | 0.000 | 0.690 | 0.150 |  |
| 2022 | null | Null |  |  | 7787.000 | 18370.000 | 360.400 | 0.000 | 0.000 | 0.825 | 0.117 |  |
| 2016 | transfer_2010_seed0 | 2010-trained DQN seed0 | 0.000 | 35000.000 | 7538.000 | 17018.000 | 365.600 | 120.000 | 0.000 | 0.000 | 0.022 | 194.467 |
| 2016 | transfer_2010_seed1 | 2010-trained DQN seed1 | 1.000 | 25000.000 | 7538.000 | 17018.000 | 365.600 | 60.000 | 0.000 | 0.000 | 0.022 | 254.467 |
| 2022 | transfer_2010_seed0 | 2010-trained DQN seed0 | 0.000 | 35000.000 | 7935.000 | 18552.000 | 356.400 | 120.000 | 0.000 | 0.000 | 0.012 | 27.681 |
| 2022 | transfer_2010_seed1 | 2010-trained DQN seed1 | 1.000 | 25000.000 | 7935.000 | 18552.000 | 356.400 | 60.000 | 0.000 | 0.000 | 0.029 | 87.681 |
