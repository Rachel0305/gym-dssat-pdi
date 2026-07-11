# 020_06 HLA2010 n-step DQN seed1 记录

## 设置

- 唯一改动：标准DQN n_steps 从1改为5。
- Seed: 1。
- Timesteps: 50000；checkpoint interval: 5000。
- reward、输入、IC、9动作、预算和其他超参数与015_12一致。
- checkpoint选择：total_reward最大；并列取最早。

## Checkpoint结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 75 | 100 | 7854 | 20876 | 0.416 | 0.016 | 322.665 |
| 10000 | 120 | 0 | 7854 | 20884 | 0.416 | 0.036 | 777.665 |
| 15000 | 120 | 0 | 7854 | 20884 | 0.416 | 0.036 | 777.665 |
| 20000 | 120 | 100 | 7854 | 20828 | 0.416 | 0.016 | 277.665 |
| 25000 | 120 | 200 | 7854 | 20851 | 0.416 | 0.016 | -222.335 |
| 30000 | 75 | 200 | 7692 | 20535 | 0.729 | 0.016 | -339.108 |
| 35000 | 60 | 50 | 7755 | 20787 | 0.542 | 0.077 | 488.941 |
| 40000 | 60 | 0 | 7632 | 20412 | 0.733 | 0.195 | 616.3 |
| 45000 | 30 | 50 | 7183 | 19866 | 0.918 | 0.072 | -53.48 |
| 50000 | 60 | 50 | 7684 | 20633 | 0.726 | 0.024 | 417.92 |

## 固定规则选点

- Step: 10000。
- Yield: 7854.0 kg/ha。
- Irrigation: 120.0 mm。
- Nitrogen: 0.0 kg/ha。
- Reward: 777.7。

- Run dir: `DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed1_020_06/nstep5_seed1_50000steps`。
- 原生WP_ET、NLCM和严格基线判定由离线评估补充。

## 正式50K原生指标与比较

| algorithm | seed | checkpoint_step | HWAM_kg_ha | WP_ET_kg_m3 | irrigation_mm | nitrogen_kg_ha | NLCM_kg_ha | training_reward | strict_success_vs_both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard DQN | 1 | 25000 | 7573 | 1.66 | 60 | 0 | 0 | 557.022 | False |
| Double DQN | 1 | 30000 | 7854 | 1.69 | 120 | 0 | 0 | 777.665 | True |
| n-step DQN | 1 | 10000 | 7854 | 1.67 | 120 | 0 | 0 | 777.665 | True |

- n-step seed1 按固定规则选中 10000 steps：HWAM=7854.0 kg/ha，WP_ET=1.67 kg/m³，I=120.0 mm，N=0.0 kg/ha，NLCM=0.0 kg/ha。
- n-step seed1 严格成功：True。
- n-step seed1 改善了标准DQN seed1，但尚未进行 n-step seed2，因此不能称跨seed稳定。
