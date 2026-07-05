# 015_14 FQ2016 baseline-relative DQN checkpoint record

## 固定设置

- year: FQ2016
- seed: 1
- timesteps: 50000
- checkpoint interval: 5000
- null baseline: 7066.0 kg/ha
- reward: terminal max(0, GWAD_final - GWAD_null) - water_cost*I - nitrogen_cost*N
- costs: water=1.0, nitrogen=5.0
- action space: I in {0,15,30}, N in {0,50,100}
- budget: I<=120, N<=300, min interval=7 days

## 基准

| scenario | GWAD kg/ha | I mm | N kg/ha |
|---|---:|---:|---:|
| null | 7066 | 0 | 0 |
| recorded_shifted | 7933 | 75 | 144 |
| DSSAT auto | 8012 | 59.9 | 0 |

## 输出

- run dir: `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps`
- summary: `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/checkpoint_summary.csv`
- daily: `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/dqn_eval_daily.csv`

## Checkpoint results

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward | irrigation_total_mgmtevent | fertilizer_total_mgmtevent | irrigation_events_mgmtevent | fertilizer_events_mgmtevent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 300 | 8012 | 14093 | 0 | 0.012 | -673.578 | 120 | 300 | 4 | 3 |
| 10000 | 120 | 300 | 8012 | 14095 | 0 | 0.012 | -673.578 | 120 | 300 | 4 | 3 |
| 15000 | 30 | 300 | 7066 | 13148 | 0.657 | 0.012 | -1529.925 | 30 | 300 | 1 | 3 |
| 20000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.075 | 0 | 0 | 0 | 0 |
| 25000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.075 | 0 | 0 | 0 | 0 |
| 30000 | 60 | 0 | 7995 | 14078 | 0.050 | 0.012 | 869.176 | 60 | 0 | 2 | 0 |
| 35000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.075 | 0 | 0 | 0 | 0 |
| 40000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.075 | 0 | 0 | 0 | 0 |
| 45000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.075 | 0 | 0 | 0 | 0 |
| 50000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.075 | 0 | 0 | 0 | 0 |

## 判断

- Best reward checkpoint: 30000, GWAD=7995.0, I=60.0, N=0.0, reward=869.2.
- Best yield checkpoint: 5000, GWAD=8012.0, I=120.0, N=300.0, reward=-673.6.