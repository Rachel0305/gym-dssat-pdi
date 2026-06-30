# 013_04 YC2014 动作窗口对照记录

## 设置

- 情景：null / recorded / dssat_auto / dqn_free_daily / dqn_agronomic_window
- RL：DQN, timesteps=500, seed=0
- 奖励：delta_grnwt - 1.0*I - 5.0*N
- 预算：I<=120.0, N<=300.0

## 汇总

| scenario | action_irrigation_total | action_fertilizer_total | mgmt_event_irrigation_total | mgmt_event_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 0 | 0 | 0 | 0 | 7825 | 17996 | 0.922 | 0.381 |
