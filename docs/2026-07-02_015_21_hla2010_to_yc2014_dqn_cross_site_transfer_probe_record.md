# 015_21 HLA2010 -> YC2014 跨站点 DQN 迁移 probe

## 目的

- 不重新训练，只迁移评估 `HLA2010 baseline-relative DQN` 到 `YC2014`。
- 判断这套 2010 训练模型是否只在 HLA 内有效，还是具备跨站点可迁移性。

## 设置

- 训练年：HLA2010
- 测试站点年份：YC2014
- 模式：smoke
- 奖励：baseline-relative，终止时 `max(0, GWAD_final - null_baseline)` 减去水/氮成本
- 动作：9-action 离散动作，I∈{0,15,30}，N∈{0,50,100}
- 约束：I<=120, N<=300, min_interval=7天

## 结果摘要

| scenario | label | train_seed | train_checkpoint_step | final_gwad | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | Null | nan | nan | 7825.000 | 0.000 | 0.000 | 0.922 | 0.381 | 7825.439 |
| recorded | Recorded expert | nan | nan | 9418.000 | 120.000 | 374.000 | 0.000 | 0.013 | 7427.902 |
| dssat_auto | DSSAT auto | nan | nan | 8713.000 | 86.500 | 0.000 | 0.000 | 0.436 | 8626.377 |
| local_dqn_best | YC2014 local DQN best | nan | nan | 9418.000 | 120.000 | 250.000 | 0.000 | 0.013 | 8048.000 |
| transfer_hla2010_seed0 | HLA2010-trained DQN seed0 | 0.000 | 35000.000 | nan | nan | nan | nan | nan | nan |

## 判读

- transfer_hla2010_seed0 未能直接迁移：Observation spaces do not match: Box(0.0, inf, (24,), float32) != Box(0.0, inf, (22,), float32)