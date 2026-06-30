# 015_05 YC2014 统一 DQN checkpoint selection seed1 复核记录

## 目的

015_04 证明 seed0 下 final model 不一定最好，checkpoint selection 能找到高产策略。本轮只将 seed 从 0 改为 1，其他设置完全不变，用于复核 YC2014 是否具备跨 seed 可复现性。

## 固定设置

- 站点年份：YC2014
- 算法：DQN, seed=1
- 总步数：50000
- checkpoint 间隔：5000
- 奖励：R_t = max(0, ΔGRNWT_t) - 1.0 × I_t - 5.0 × N_t
- 动作空间：9 个离散动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha
- 预算：I≤120 mm，N≤300 kg/ha
- 最小操作间隔：7 days
- 管理模式：IRRIG=L, FERTI=L

## 输出文件

- 日值总表：`DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/seed1/015_05_yc2014_unified_dqn_checkpoint_seed1_daily.csv`
- 汇总表：`DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/seed1/015_05_yc2014_unified_dqn_checkpoint_seed1_summary.csv`
- 对比图：`DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/seed1/figures/yc2014_unified_dqn_checkpoint_seed1.png`

## seed1 汇总结果

| seed | checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5000 | 45 | 300 | 9126 | 20176 | 0.514 | 0.013 | 7580.893 |
| 1 | 10000 | 120 | 300 | 9418 | 20493 | 0 | 0.013 | 7797.902 |
| 1 | 15000 | 120 | 300 | 9416 | 20462 | 0 | 0.013 | 7796.277 |
| 1 | 20000 | 120 | 300 | 9417 | 20487 | 0 | 0.013 | 7796.998 |
| 1 | 25000 | 120 | 300 | 9418 | 20493 | 0 | 0.013 | 7797.902 |
| 1 | 30000 | 30 | 300 | 9219 | 19886 | 0 | 0.082 | 7689.161 |
| 1 | 35000 | 120 | 300 | 8939 | 19556 | 0 | 0.082 | 7319.233 |
| 1 | 40000 | 120 | 300 | 8939 | 19556 | 0 | 0.082 | 7319.233 |
| 1 | 45000 | 120 | 300 | 8939 | 19556 | 0 | 0.082 | 7319.233 |
| 1 | 50000 | 60 | 300 | 8749 | 19250 | 0.530 | 0.082 | 7188.980 |

## 关键判断

- seed1 最高产量 checkpoint：10000，产量=9418.0 kg/ha，I=120.0 mm，N=300.0 kg/ha。
- seed1 最高 reward checkpoint：10000，reward=7797.9，产量=9418.0 kg/ha。
- seed1 final checkpoint：50000，产量=8749.0 kg/ha，I=60.0 mm，N=300.0 kg/ha。

## 与 seed0 best checkpoint 对比

- seed0 best：checkpoint=5000，产量=9418.0 kg/ha，I=120.0 mm，N=250.0 kg/ha。
- seed1 best：checkpoint=10000，产量=9418.0 kg/ha，I=120.0 mm，N=300.0 kg/ha。

## 初步结论

- 如果 seed1 best 也接近 9400 kg/ha 且包含合理施氮，则 YC2014 可以进入正式 DQN 成功案例候选。
- 如果 seed1 best 明显低于 seed0，则说明 seed0 仍可能存在偶然性，需要继续 seed2 或调整训练机制。
