# 011_09 HLA 2010 single-variable PPO 5k diagnosis

## 背景

011_07/011_08 的水氮联合 budgeted daily PPO 已经跑通，但两个 seed 都表现出“早期使用预算”的倾向：

- N150 在 DAP 2/9/16 用完；
- I120 也在早期用完或较早分散用完。

用户指出：此前只优化氮或只优化灌溉时，PPO 的施肥/灌溉结果曾经更合理。

## 本轮目标

复查单变量 PPO：

1. irrigation-only PPO；
2. fertilization-only PPO。

目的：

判断“早期打满预算”是否只出现在水氮联合动作空间中，还是单变量也会出现。

## 设置

共同设置：

- year: HLA 2010
- seed: 0
- timesteps: 5000
- linked management:
  - irrigation-only: `IRRIG=L, FERTI=R`
  - fertilization-only: `IRRIG=R, FERTI=L`
- reward:
  - irrigation-only: official `irrigation_reward`
  - fertilization-only: official `fertilization_reward`
- daily interaction retained.

预算/安全约束：

- irrigation-only:
  - `I_total <= 120 mm`
  - `I_day <= 30 mm`
  - min interval = 7 days
- fertilization-only:
  - `N_total <= 150 kg/ha`
  - `N_day <= 50 kg/ha`
  - min interval = 7 days

## 输出

目录仍在：

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/`

命名：

- `irrigation_seed0_5000steps`
- `fertilization_seed0_5000steps`

比较重点：

- 是否完成；
- 是否在预算内；
- 管理时序是否仍然早期打满；
- 产量与 null、joint PPO 的差异。

## Result

Both single-variable runs completed.

| run | HWAM | irrigation | nitrogen | management timing |
| --- | ---: | ---: | ---: | --- |
| joint seed0 | 7854 | 120 | 150 | DAP 2/9/16 N50+I30, DAP23 I30 |
| joint seed1 | 7854 | 120 | 150 | DAP 2/9/16 N50, irrigation spread DAP2-44 |
| irrigation-only | 7854 | 120 | 0 | DAP 2/9/16/23 I30 |
| fertilization-only | 6971 | 0 | 150 | DAP 2/9/16 N50 |

Conclusion: single-variable PPO also uses the available budget early. The tendency is therefore not unique to joint water-nitrogen action space.

HLA 2010 response is mainly irrigation-driven: irrigation-only reaches the same yield as joint PPO, while fertilization-only barely improves over null.
