# 020_04 HLA2010 Double DQN seed1 严格单变量对照记录

## 设置

- 唯一改动：标准DQN TD target改为Double DQN TD target。
- Seed: 1。
- Timesteps: 50000；checkpoint interval: 5000。
- HLA2010 null baseline: 6956.0 kg/ha。
- reward、输入、IC、9动作、预算和所有其他超参数与015_12 seed1相同。
- 选择规则：total_reward最大；并列时取最早checkpoint。

## Checkpoint结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 120 | 300 | 7854 | 20812 | 0.416 | 0.016 | -722.335 |
| 10000 | 120 | 300 | 7854 | 20645 | 0.416 | 0.016 | -722.335 |
| 15000 | 120 | 300 | 7854 | 20746 | 0.416 | 0.016 | -722.335 |
| 20000 | 120 | 50 | 7854 | 20834 | 0.416 | 0.016 | 527.665 |
| 25000 | 45 | 300 | 7854 | 20886 | 0.435 | 0.016 | -647.335 |
| 30000 | 120 | 0 | 7854 | 20825 | 0.416 | 0.035 | 777.665 |
| 35000 | 60 | 200 | 7483 | 20165 | 0.776 | 0.016 | -532.879 |
| 40000 | 120 | 300 | 7854 | 20886 | 0.416 | 0.016 | -722.335 |
| 45000 | 120 | 100 | 7854 | 20879 | 0.416 | 0.016 | 277.665 |
| 50000 | 45 | 300 | 7830 | 20863 | 0.416 | 0.016 | -670.519 |

## 固定规则选点

- Step: 30000。
- Yield: 7854.0 kg/ha。
- Irrigation: 120.0 mm。
- Nitrogen: 0.0 kg/ha。
- Reward: 777.7。

- Run dir: `DSSAT_auto_validation/HLA_2004/hla2010_double_dqn_seed1_020_04/double_dqn_seed1_50000steps`。
- 原生WP_ET/NLCM和严格基线判定由离线评估补充。

## 正式50K结果与严格比较

| algorithm | seed | checkpoint_step | HWAM_kg_ha | WP_ET_kg_m3 | irrigation_mm | nitrogen_kg_ha | NLCM_kg_ha | training_reward | strict_success_vs_both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard DQN | 1 | 25000 | 7573 | 1.66 | 60 | 0 | 0 | 557.022 | False |
| Double DQN | 1 | 30000 | 7854 | 1.69 | 120 | 0 | 0 | 777.665 | True |

- Double DQN 按固定规则选中 30000 steps：HWAM=7854.0 kg/ha，WP_ET=1.69 kg/m³，I=120.0 mm，N=0.0 kg/ha，NLCM=0.0 kg/ha。
- Double DQN 严格成功：True。
- 与标准DQN seed1相比：产量变化 +281.0 kg/ha，灌溉变化 +60.0 mm，施氮变化 +0.0 kg/ha。
- 全部checkpoint动作与MgmtEvent总量最大差异：灌溉 0.000 mm，施氮 0.000 kg/ha。
- Double DQN与标准DQN seed1输入文件一致：True。
- 结论：当前框架下 Double DQN 使 seed1 达到与标准DQN seed0相同的严格成功点，但这只是一个 seed 的算法对照，尚不能证明 Double DQN 已跨seed稳定。
