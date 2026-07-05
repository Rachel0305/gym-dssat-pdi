# 015_12 HLA2015 baseline-relative DQN checkpoint record

## Settings

- Year: HLA2015
- Seed: 1
- Timesteps: 50000
- Checkpoint interval: 5000
- Null baseline yield: 6486.0 kg/ha
- Reward: terminal max(0, GWAD_final - GWAD_null_site_year) minus water/nitrogen costs
- Costs: water=1.0, nitrogen=5.0
- Action space: 9-action, I in {0,15,30}, N in {0,50,100}
- Budget: I<=120, N<=300, minimum interval=7 days

## Outputs

- Run dir: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps`
- Summary: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps/checkpoint_summary.csv`
- Daily: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps/dqn_eval_daily.csv`
- Figure: `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps/figures/hla2015_baseline_relative_seed1_50000steps_checkpoint_diagnostic.png`

## Checkpoint Results

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 300 | 7649 | 19045 | 0 | 0.015 | -456.593 |
| 10000 | 120 | 300 | 7653 | 19081 | 0 | 0.015 | -452.813 |
| 15000 | 120 | 300 | 7647 | 18942 | 0 | 0.015 | -458.811 |
| 20000 | 120 | 300 | 7653 | 19081 | 0 | 0.015 | -452.813 |
| 25000 | 45 | 150 | 7653 | 19081 | 0 | 0.015 | 372.187 |
| 30000 | 45 | 150 | 7653 | 19081 | 0 | 0.015 | 372.187 |
| 35000 | 120 | 200 | 7652 | 19049 | 0 | 0.015 | 45.833 |
| 40000 | 75 | 0 | 7653 | 19077 | 0 | 0.076 | 1092.096 |
| 45000 | 120 | 100 | 7653 | 19081 | 0 | 0.015 | 547.187 |
| 50000 | 45 | 50 | 7653 | 19081 | 0 | 0.015 | 872.187 |

## Key Judgement

- Best yield checkpoint: 10000, yield=7653.0, I=120.0, N=300.0, reward=-452.8.
- Best reward checkpoint: 40000, yield=7653.0, I=75.0, N=0.0, reward=1092.1.
- Final checkpoint: 50000, yield=7653.0, I=45.0, N=50.0, reward=872.2.