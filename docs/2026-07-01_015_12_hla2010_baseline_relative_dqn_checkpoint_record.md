# 015_12 HLA2010 baseline-relative DQN checkpoint record

## Settings

- Year: HLA2010
- Seed: 1
- Timesteps: 50000
- Checkpoint interval: 5000
- Null baseline yield: 6956.0 kg/ha
- Reward: terminal max(0, GWAD_final - GWAD_null_site_year) minus water/nitrogen costs
- Costs: water=1.0, nitrogen=5.0
- Action space: 9-action, I in {0,15,30}, N in {0,50,100}
- Budget: I<=120, N<=300, minimum interval=7 days

## Outputs

- Run dir: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps`
- Summary: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/checkpoint_summary.csv`
- Daily: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/dqn_eval_daily.csv`
- Figure: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/figures/hla2010_baseline_relative_seed1_50000steps_checkpoint_diagnostic.png`

## Checkpoint Results

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 300 | 7854 | 20846 | 0.416 | 0.016 | -722.335 |
| 10000 | 45 | 300 | 7660 | 20342 | 0.532 | 0.016 | -840.993 |
| 15000 | 15 | 100 | 7351 | 19905 | 0.885 | 0.016 | -120.384 |
| 20000 | 60 | 0 | 7483 | 20165 | 0.776 | 0.162 | 467.121 |
| 25000 | 60 | 0 | 7573 | 20255 | 0.751 | 0.173 | 557.022 |
| 30000 | 0 | 0 | 6956 | 19344 | 0.919 | 0.157 | 0.454 |
| 35000 | 60 | 0 | 7483 | 20165 | 0.776 | 0.168 | 467.121 |
| 40000 | 0 | 0 | 6956 | 19344 | 0.919 | 0.157 | 0.454 |
| 45000 | 105 | 0 | 7483 | 20165 | 0.776 | 0.113 | 422.121 |
| 50000 | 0 | 0 | 6956 | 19344 | 0.919 | 0.157 | 0.454 |

## Key Judgement

- Best yield checkpoint: 5000, yield=7854.0, I=120.0, N=300.0, reward=-722.3.
- Best reward checkpoint: 25000, yield=7573.0, I=60.0, N=0.0, reward=557.0.
- Final checkpoint: 50000, yield=6956.0, I=0.0, N=0.0, reward=0.5.