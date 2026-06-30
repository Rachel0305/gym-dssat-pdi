# 015_01 YC2014 统一 DQN 正式训练 seed0 记录

## 设置

- 站点年份：Yucheng 2014
- 情景：`dqn_unified_9action_free_daily`
- 算法：DQN, timesteps=50000, seed=0
- 奖励：`max(0, ΔGRNWT) - 1.0 × irrigation - 5.0 × nitrogen`
- 动作空间：9 动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha
- 预算：I≤120 mm，N≤300 kg/ha
- 最小操作间隔：7 days
- 管理模式：IRRIG=L, FERTI=L，保留原管理表/指针以支持动态动作写入

## 输出

- 日值：`DSSAT_auto_validation\yc2014_unified_dqn_formal_015_01\seed0\015_01_yc2014_unified_dqn_50k_seed0_daily.csv`
- 汇总：`DSSAT_auto_validation\yc2014_unified_dqn_formal_015_01\seed0\015_01_yc2014_unified_dqn_50k_seed0_summary.csv`

## 结果汇总

| scenario | action_irrigation_total | action_fertilizer_total | mgmt_event_irrigation_total | mgmt_event_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn_unified_9action_free_daily | 90 | 0 | 90 | 0 | 8056 | 18227 | 0.743 | 0.398 | 7966.013 |

## 初步判读

- 本轮为统一正式框架的第一个 seed0 长训练结果。
- 本轮统一 reward 与所有后续站点保持一致，不能再为不同站点单独更换 reward。
- 是否进入正式论文结果，还需要与 null / recorded / DSSAT auto 统一图表对比，并继续 seed1/seed2 稳定性验证。
- 初步结果显示，统一 reward 下 seed0 策略使用 90 mm 灌溉、0 kg/ha 施氮，产量 8056 kg/ha。该结果明显优于 null 的 7825 kg/ha，但低于此前 YC2014 DQN/recorded expert 约 9417-9418 kg/ha 的水平。
- 这说明当前统一 reward 并未复现此前“高产 + 合理水氮操作”的效果，主要问题可能是 `N_COST=5` 对施氮惩罚偏强，或阶段 `ΔGRNWT` 奖励无法及时反映施氮的延迟收益。
- 因此本轮结果应作为正式框架的首个诊断结果，而不是最终成功结果。下一步应先做低成本 reward/cost 诊断，而不是直接扩 seed 或扩站点。
