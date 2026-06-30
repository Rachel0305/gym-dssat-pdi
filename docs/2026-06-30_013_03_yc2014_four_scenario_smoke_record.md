# 013_03 YC2014 四情景 smoke 记录

## 设置

- 四情景：null / recorded / dssat_auto / dqn_economic_smoke
- RL：DQN, timesteps=500, seed=0
- 奖励：delta_grnwt - 1.0*I - 5.0*N
- 预算：I<=120.0, N<=300.0

## 汇总

| scenario | event_irrigation_total | event_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| null | 0 | 0 | 7825 | 17996 | 0.922 | 0.381 | nan |
| recorded | 120 | 374 | 9418 | 20514 | 0 | 0.013 | nan |
| dssat_auto | 86.500 | 0 | 8713 | 18945 | 0 | 0.436 | nan |
| dqn_economic_smoke | 120 | 300 | 7825 | 17996 | 0.922 | 0.381 | 6205.439 |
