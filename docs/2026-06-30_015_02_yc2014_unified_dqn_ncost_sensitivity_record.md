# 015_02 YC2014 统一 DQN 氮成本敏感性诊断记录

## 目的

015_01 的统一 DQN 50K seed0 在 YC2014 上只灌水、不施氮。本轮用 5K 低成本训练扫描统一奖励函数中的氮成本系数，判断是否是 N_COST=5.0 过强导致施氮被压制。

## 固定设置

- 站点年份：YC2014
- DQN timesteps=5000, seed=0
- 动作空间：9 个离散动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha
- 预算：I≤120 mm，N≤300 kg/ha
- 最小操作间隔：7 days
- 水成本：WATER_COST=1.0
- 管理模式：IRRIG=L, FERTI=L

## 扫描变量

- N_COST = 0.5, 1.0, 2.0, 3.0, 5.0
- 奖励：R_t = max(0, ΔGRNWT_t) - 1.0 × I_t - N_COST × N_t

## 输出文件

- 日值总表：`DSSAT_auto_validation/yc2014_unified_dqn_ncost_sensitivity_015_02/seed0/015_02_yc2014_unified_dqn_ncost_sensitivity_daily.csv`
- 汇总表：`DSSAT_auto_validation/yc2014_unified_dqn_ncost_sensitivity_015_02/seed0/015_02_yc2014_unified_dqn_ncost_sensitivity_summary.csv`
- 对比图：`DSSAT_auto_validation/yc2014_unified_dqn_ncost_sensitivity_015_02/seed0/figures/yc2014_unified_dqn_ncost_sensitivity.png`

## 汇总结果

| scenario | action_irrigation_total | action_fertilizer_total | mgmt_event_irrigation_total | mgmt_event_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward | n_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn_unified_ncost_0p5 | 120 | 250 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 9172.855 | 0.500 |
| dqn_unified_ncost_1p0 | 120 | 250 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 9047.855 | 1 |
| dqn_unified_ncost_2p0 | 120 | 250 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 8797.855 | 2 |
| dqn_unified_ncost_3p0 | 120 | 250 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 8547.855 | 3 |
| dqn_unified_ncost_5p0 | 120 | 250 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 8047.855 | 5 |

## 初步结论

- 最高产量候选：N_COST=0.5，产量=9418.0 kg/ha，I=120.0 mm，N=250.0 kg/ha。
- 最高累积 reward 候选：N_COST=0.5，reward=9172.9，I=120.0 mm，N=250.0 kg/ha。
- 本轮是奖励函数诊断，不是正式论文训练结果；不能据此直接宣布 DQN 成功或失败。
- 如果低 N_COST 恢复施氮且产量明显上升，说明正式统一奖励应重新校准氮成本；如果所有 N_COST 仍不施氮，则需要改奖励结构而不是继续微调成本。
