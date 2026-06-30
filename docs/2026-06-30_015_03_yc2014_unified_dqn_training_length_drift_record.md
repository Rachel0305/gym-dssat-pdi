# 015_03 YC2014 统一 DQN 训练步长漂移诊断记录

## 目的

015_02 显示 N_COST=5 在 5K 下可以学到 I120/N250 高产策略，但 015_01 的 50K 结果变成只灌水、不施氮。本轮复用已有 5K/50K 结果，只新跑 10K 和 20K，判断是否存在长训练策略漂移。

## 固定设置

- 站点年份：YC2014
- 算法：DQN, seed=0
- 奖励：R_t = max(0, ΔGRNWT_t) - 1.0 × I_t - 5.0 × N_t
- 动作空间：9 个离散动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha
- 预算：I≤120 mm，N≤300 kg/ha
- 最小操作间隔：7 days
- 管理模式：IRRIG=L, FERTI=L

## 输出文件

- 日值总表：`DSSAT_auto_validation/yc2014_unified_dqn_training_length_drift_015_03/seed0/015_03_yc2014_unified_dqn_training_length_drift_daily.csv`
- 汇总表：`DSSAT_auto_validation/yc2014_unified_dqn_training_length_drift_015_03/seed0/015_03_yc2014_unified_dqn_training_length_drift_summary.csv`
- 对比图：`DSSAT_auto_validation/yc2014_unified_dqn_training_length_drift_015_03/seed0/figures/yc2014_unified_dqn_training_length_drift.png`

## 汇总结果

| timesteps | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 8047.855 |
| 10000 | 15 | 300 | 8749 | 19250 | 0.530 | 0.082 | 7233.970 |
| 20000 | 120 | 100 | 8939 | 19556 | 0 | 0.082 | 8319.233 |
| 50000 | 90 | 0 | 8056 | 18227 | 0.743 | 0.398 | 7966.013 |

## 初步结论

- 如果 10K/20K 与 5K 保持 I120/N250，而 50K 仍是 I90/N0，则优先怀疑 015_01 配置与本轮不一致，或 50K 单次运行存在训练偶然性。
- 如果 10K/20K 逐步减少施氮，则支持长训练策略漂移假设。
- 本轮仍然是诊断，不是正式论文结果；下一步应根据漂移模式决定是复查脚本配置，还是调整 DQN 训练机制。
