# 015_18 HLA2010-trained DQN transfer evaluation on HLA2007

## Purpose

Evaluate whether DQN checkpoints trained on HLA2010 can produce reasonable water-nitrogen decisions on HLA2007 without additional training.

## Tested models

- HLA2010 seed0 checkpoint 35000 -> HLA2007
- HLA2010 seed1 checkpoint 25000 -> HLA2007

## Outputs

- Daily CSV: `DSSAT_auto_validation/HLA_2004/hla2010_to_2007_dqn_transfer_eval_015_18/hla2010_to_2007_transfer_eval_daily.csv`
- Summary CSV: `DSSAT_auto_validation/HLA_2004/hla2010_to_2007_dqn_transfer_eval_015_18/hla2010_to_2007_transfer_eval_summary.csv`
- Figure: `DSSAT_auto_validation/HLA_2004/hla2010_to_2007_dqn_transfer_eval_015_18/figures/hla2010_to_2007_transfer_eval_process.png`

## Results

| scenario | label | train_year | train_seed | train_checkpoint_step | final_gwad | final_cwad | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | Null |  |  |  | 7830.000 | 19528.000 | 0.000 | 0.000 | 0.895 | 0.129 |  |
| recorded | Recorded expert |  |  |  | 7986.000 | 19809.000 | 30.000 | 165.000 | 0.000 | 0.014 |  |
| dssat_auto | DSSAT auto |  |  |  | 7973.000 | 19638.000 | 93.500 | 0.000 | 0.000 | 0.032 |  |
| transfer_2010_seed0 | 2010-trained DQN seed0 | 2010.000 | 0.000 | 35000.000 | 7987.000 | 19816.000 | 120.000 | 0.000 | 0.000 | 0.059 | 36.733 |
| transfer_2010_seed1 | 2010-trained DQN seed1 | 2010.000 | 1.000 | 25000.000 | 7987.000 | 19786.000 | 45.000 | 0.000 | 0.153 | 0.103 | 111.733 |

## Interpretation

- If transfer models are close to local DQN, the policy has useful cross-year transfer.
- If transfer models are closer to null or use resources poorly, current DQN is mainly year-specific.