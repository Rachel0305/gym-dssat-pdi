# 013_06 YC2014 linked DQN smoke 记录

## 设置

- 情景：null / recorded / dssat_auto / dqn_linked_free_daily / dqn_linked_agronomic_window
- RL：DQN, timesteps=500, seed=0
- 奖励：delta_grnwt - 1.0*I - 5.0*N
- 预算：I<=120.0, N<=300.0
- 关键修复：DQN 输入保留原始管理表/指针，只设置 IRRIG=L, FERTI=L，确保动作进入 MgmtEvent.OUT。

## 汇总

| scenario | action_irrigation_total | action_fertilizer_total | mgmt_event_irrigation_total | mgmt_event_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 0 | 0 | 0 | 0 | 7825 | 17996 | 0.922 | 0.381 | nan |
| recorded | 0 | 0 | 120 | 374 | 9418 | 20514 | 0 | 0.013 | nan |
| dssat_auto | 0 | 0 | 86.500 | 0 | 8713 | 18945 | 0 | 0.436 | nan |
| dqn_linked_free_daily | 120 | 300 | 120 | 300 | 9418 | 20504 | 0 | 0.013 | 7797.607 |
| dqn_linked_agronomic_window | 120 | 300 | 120 | 300 | 9416 | 20420 | 0 | 0.013 | 7796.431 |
