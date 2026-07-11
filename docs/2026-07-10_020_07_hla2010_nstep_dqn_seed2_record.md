# 020_07 HLA2010 n-step DQN seed2 记录

- 唯一改动：标准DQN n_steps从1改为5。
- Seed: 2。
- Timesteps: 50000；checkpoint interval: 5000。
- checkpoint选择：total_reward最大；并列取最早。

## Checkpoint结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 100 | 7854 | 20886 | 0.416 | 0.016 | 277.665 |
| 10000 | 120 | 50 | 7854 | 20886 | 0.416 | 0.016 | 527.665 |
| 15000 | 120 | 100 | 7854 | 20852 | 0.416 | 0.016 | 277.665 |
| 20000 | 120 | 0 | 7854 | 20852 | 0.416 | 0.039 | 777.665 |
| 25000 | 120 | 150 | 7854 | 20860 | 0.416 | 0.016 | 27.665 |
| 30000 | 120 | 100 | 7854 | 20876 | 0.416 | 0.016 | 277.665 |
| 35000 | 120 | 50 | 7854 | 20856 | 0.416 | 0.016 | 527.665 |
| 40000 | 120 | 50 | 7854 | 20828 | 0.416 | 0.016 | 527.665 |
| 45000 | 120 | 200 | 7854 | 20828 | 0.416 | 0.016 | -222.335 |
| 50000 | 120 | 100 | 7854 | 20828 | 0.416 | 0.016 | 277.665 |

## 固定规则选点

- Step: 20000；Yield: 7854.0 kg/ha；I: 120.0 mm；N: 0.0 kg/ha；Reward: 777.7。
- Run dir: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed2_020_07/nstep5_seed2_50000steps`。
- 原生WP_ET、NLCM和严格基线判定由离线评估补充。

## 正式50K原生指标与比较

| algorithm | seed | checkpoint_step | HWAM_kg_ha | WP_ET_kg_m3 | irrigation_mm | nitrogen_kg_ha | NLCM_kg_ha | training_reward | strict_success_vs_both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard DQN | 1 | 25000 | 7573 | 1.66 | 60 | 0 | 0 | 557.022 | False |
| Double DQN | 1 | 30000 | 7854 | 1.69 | 120 | 0 | 0 | 777.665 | True |
| n-step DQN | 1 | 10000 | 7854 | 1.67 | 120 | 0 | 0 | 777.665 | True |
| n-step DQN | 2 | 20000 | 7854 | 1.68 | 120 | 0 | 0 | 777.665 | True |

- n-step seed2 按固定规则选中 20000 steps：HWAM=7854.0 kg/ha，WP_ET=1.68 kg/m³，I=120.0 mm，N=0.0 kg/ha，NLCM=0.0 kg/ha。
- n-step seed2 严格成功：True。
- n-step seed1和seed2均通过严格基线判定，说明n-step在当前HLA2010框架下出现初步跨seed稳定迹象，但仍只验证了两个seed。
