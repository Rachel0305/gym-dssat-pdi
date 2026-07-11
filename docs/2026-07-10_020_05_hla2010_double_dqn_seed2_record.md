# 020_05 HLA2010 Double DQN seed2 记录

## 设置

- 唯一算法改动：标准DQN TD target改为Double DQN TD target。
- Seed: 2。
- Timesteps: 50000；checkpoint interval: 5000。
- reward、输入、IC、9动作、预算和其他超参数与015_12/020_04一致。
- checkpoint选择：total_reward最大；并列取最早。

## Checkpoint结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 45 | 200 | 7778 | 20810 | 0.63 | 0.016 | -222.881 |
| 10000 | 120 | 300 | 7854 | 20842 | 0.416 | 0.016 | -722.335 |
| 15000 | 120 | 200 | 7854 | 20674 | 0.416 | 0.016 | -222.335 |
| 20000 | 120 | 300 | 7854 | 20718 | 0.416 | 0.016 | -722.335 |
| 25000 | 45 | 250 | 7457 | 20139 | 0.776 | 0.016 | -794.234 |
| 30000 | 15 | 300 | 7293 | 19906 | 0.842 | 0.016 | -1177.736 |
| 35000 | 60 | 200 | 7457 | 20139 | 0.776 | 0.016 | -558.829 |
| 40000 | 0 | 100 | 6962 | 19355 | 0.918 | 0.016 | -493.854 |
| 45000 | 0 | 0 | 6956 | 19344 | 0.919 | 0.157 | 0.454 |
| 50000 | 30 | 200 | 7573 | 20255 | 0.751 | 0.016 | -412.978 |

## 固定规则选点

- Step: 45000。
- Yield: 6956.0 kg/ha。
- Irrigation: 0.0 mm。
- Nitrogen: 0.0 kg/ha。
- Reward: 0.5。

- Run dir: `DSSAT_auto_validation/HLA_2004/hla2010_double_dqn_seed2_020_05/double_dqn_seed2_50000steps`。
- 原生WP_ET、NLCM和严格基线判定由离线评估补充。

## 正式50K原生指标与比较

| algorithm | seed | checkpoint_step | HWAM_kg_ha | WP_ET_kg_m3 | irrigation_mm | nitrogen_kg_ha | NLCM_kg_ha | training_reward | strict_success_vs_both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard DQN | 1 | 25000 | 7573 | 1.66 | 60 | 0 | 0 | 557.022 | False |
| Double DQN | 1 | 30000 | 7854 | 1.69 | 120 | 0 | 0 | 777.665 | True |
| Double DQN | 2 | 45000 | 6956 | 1.63 | 0 | 0 | 0 | 0.454 | False |

- Double DQN seed2 按固定规则选中 45000 steps：HWAM=6956.0 kg/ha，WP_ET=1.63 kg/m³，I=0.0 mm，N=0.0 kg/ha，NLCM=0.0 kg/ha。
- Double DQN seed2 严格成功：False。
- 结论：Double DQN seed1 的改善没有在 seed2 复现；当前只能报告单seed算法改善，不能称跨seed稳定。
