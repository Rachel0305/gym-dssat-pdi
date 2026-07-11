# 020_08 HLA2010 n-step DQN seed0 记录

- 唯一改动：标准DQN n_steps从1改为5。
- Seed: 0。
- Timesteps: 50000；checkpoint interval: 5000。
- checkpoint选择：total_reward最大；并列取最早。

## Checkpoint结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 300 | 7660 | 20342 | 0.532 | 0.016 | -915.993 |
| 10000 | 30 | 50 | 7457 | 20139 | 0.776 | 0.038 | 221.171 |
| 15000 | 0 | 0 | 6956 | 19344 | 0.919 | 0.157 | 0.454 |
| 20000 | 30 | 0 | 7457 | 20139 | 0.776 | 0.2 | 471.171 |
| 25000 | 45 | 50 | 7457 | 20139 | 0.776 | 0.03 | 206.171 |
| 30000 | 75 | 0 | 7854 | 20886 | 0.416 | 0.1 | 822.665 |
| 35000 | 75 | 50 | 7792 | 20824 | 0.416 | 0.053 | 510.752 |
| 40000 | 105 | 50 | 7854 | 20880 | 0.416 | 0.016 | 542.665 |
| 45000 | 30 | 0 | 7457 | 20139 | 0.776 | 0.213 | 471.171 |
| 50000 | 30 | 0 | 7457 | 20139 | 0.776 | 0.196 | 471.171 |

## 固定规则选点

- Step: 30000；Yield: 7854.0 kg/ha；I: 75.0 mm；N: 0.0 kg/ha；Reward: 822.7。
- Run dir: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed0_020_08/nstep5_seed0_50000steps`。
- 原生WP_ET、NLCM和严格基线判定由离线评估补充。

## 正式50K原生指标与三seed比较

| algorithm | seed | checkpoint_step | HWAM_kg_ha | WP_ET_kg_m3 | irrigation_mm | nitrogen_kg_ha | NLCM_kg_ha | training_reward | strict_success_vs_both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard DQN | 0 | 15000 | 7854 | 1.69 | 120 | 0 | 0 | 777.665 | True |
| n-step DQN | 0 | 30000 | 7854 | 1.69 | 75 | 0 | 0 | 822.665 | True |
| n-step DQN | 1 | 10000 | 7854 | 1.67 | 120 | 0 | 0 | 777.665 | True |
| n-step DQN | 2 | 20000 | 7854 | 1.68 | 120 | 0 | 0 | 777.665 | True |

- n-step seed0 按固定规则选中 30000 steps：HWAM=7854.0 kg/ha，WP_ET=1.69 kg/m³，I=75.0 mm，N=0.0 kg/ha，NLCM=0.0 kg/ha。
- n-step seed0 严格成功：True。
- n-step 三个 seed 严格成功数：3/3。
- 结论：n-step 在当前HLA2010框架下完成三seed严格成功，成为目前最有稳定性证据的算法配置。
