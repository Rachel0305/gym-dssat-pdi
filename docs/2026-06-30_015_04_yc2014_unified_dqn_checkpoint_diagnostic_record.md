# 015_04 YC2014 统一 DQN checkpoint 诊断记录

## 目的

015_03 表明同一 DQN 框架在不同训练步长下策略不稳定。本轮在同一次 50K 训练中每隔 5K 评估 checkpoint，判断是否应该使用 best checkpoint 而不是 final model。

## 固定设置

- 站点年份：YC2014
- 算法：DQN, seed=0
- 总步数：50000
- checkpoint 间隔：5000
- 奖励：R_t = max(0, ΔGRNWT_t) - 1.0 × I_t - 5.0 × N_t
- 动作空间：9 个离散动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha
- 预算：I≤120 mm，N≤300 kg/ha
- 最小操作间隔：7 days
- 管理模式：IRRIG=L, FERTI=L

## 输出文件

- 日值总表：`DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/seed0/015_04_yc2014_unified_dqn_checkpoint_daily.csv`
- 汇总表：`DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/seed0/015_04_yc2014_unified_dqn_checkpoint_summary.csv`
- 对比图：`DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/seed0/figures/yc2014_unified_dqn_checkpoint_diagnostic.png`

## 汇总结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 8047.855 |
| 10000 | 120 | 300 | 9418 | 20512 | 0 | 0.013 | 7797.834 |
| 15000 | 120 | 300 | 8939 | 19555 | 0 | 0.082 | 7318.969 |
| 20000 | 120 | 300 | 9417 | 20491 | 0 | 0.032 | 7797.288 |
| 25000 | 75 | 300 | 8939 | 19556 | 0 | 0.082 | 7364.233 |
| 30000 | 120 | 100 | 8939 | 19280 | 0 | 0.204 | 8319.233 |
| 35000 | 90 | 200 | 8939 | 19365 | 0 | 0.162 | 7849.233 |
| 40000 | 120 | 200 | 8939 | 19314 | 0 | 0.193 | 7819.233 |
| 45000 | 90 | 100 | 8939 | 19264 | 0 | 0.225 | 8349.233 |
| 50000 | 120 | 100 | 8939 | 19208 | 0 | 0.305 | 8319.233 |

## 关键判断

- 最高产量 checkpoint：5000，产量=9418.0 kg/ha，I=120.0 mm，N=250.0 kg/ha。
- 最高 reward checkpoint：45000，reward=8349.2，产量=8939.0 kg/ha。
- final checkpoint：50000，产量=8939.0 kg/ha，I=120.0 mm，N=100.0 kg/ha。
- 如果最高产量或最高 reward 明显早于 final，则正式训练应采用 checkpoint selection，而不是直接取最终模型。
