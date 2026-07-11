# 020_02 HLA2010 seed2 稳定性复核记录

## 固定设置

- Seed: 2
- Timesteps: 50000
- Checkpoint interval: 5000
- Null baseline yield: 6956.0 kg/ha
- 算法、reward、动作、预算、IC 和输入链路均复用 015_12。
- checkpoint 选择规则：total_reward 最大；并列时取最早 checkpoint。
- 本记录不允许依据产量或图形人工改选 checkpoint。

## 输出

- Run dir: `DSSAT_auto_validation/HLA_2004/hla2010_seed2_stability_020_02/2010/baseline_relative_seed2_50000steps`
- Summary: `DSSAT_auto_validation/HLA_2004/hla2010_seed2_stability_020_02/2010/baseline_relative_seed2_50000steps/checkpoint_summary.csv`
- Daily: `DSSAT_auto_validation/HLA_2004/hla2010_seed2_stability_020_02/2010/baseline_relative_seed2_50000steps/dqn_eval_daily.csv`

## Checkpoint 结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 90 | 300 | 7854 | 20886 | 0.416 | 0.016 | -692.335 |
| 10000 | 120 | 300 | 7854 | 20839 | 0.416 | 0.016 | -722.335 |
| 15000 | 120 | 300 | 7854 | 20879 | 0.416 | 0.016 | -722.335 |
| 20000 | 0 | 200 | 6962 | 19355 | 0.918 | 0.016 | -994.055 |
| 25000 | 45 | 100 | 7483 | 20165 | 0.776 | 0.016 | -17.879 |
| 30000 | 60 | 0 | 7660 | 20342 | 0.532 | 0.183 | 644.007 |
| 35000 | 15 | 0 | 7308 | 19943 | 0.842 | 0.211 | 336.811 |
| 40000 | 0 | 0 | 6956 | 19344 | 0.919 | 0.157 | 0.454 |
| 45000 | 90 | 150 | 7457 | 20139 | 0.776 | 0.016 | -338.829 |
| 50000 | 60 | 50 | 7457 | 20139 | 0.776 | 0.018 | 191.171 |

## 预先固定规则选中的 checkpoint

- Step: 30000
- Yield: 7660.0 kg/ha
- Irrigation: 60.0 mm
- Nitrogen: 0.0 kg/ha
- Reward: 644.0

DSSAT 原生 WP_ET、NLCM 及相对基线的严格成功判定由后续离线评估脚本补充。

## DSSAT 原生指标与最终判定

| seed | checkpoint_step | HWAM_kg_ha | WP_ET_kg_m3 | irrigation_mm | nitrogen_kg_ha | NLCM_kg_ha | training_reward | strict_success_vs_both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 15000 | 7854 | 1.69 | 120 | 0 | 0 | 777.665 | True |
| 1 | 25000 | 7573 | 1.66 | 60 | 0 | 0 | 557.022 | False |
| 2 | 30000 | 7660 | 1.66 | 60 | 0 | 0 | 644.007 | False |

- seed2 固定规则选点：30000 steps。
- seed2 原生指标：HWAM=7660.0 kg/ha，WP_ET=1.66 kg/m³，I=60.0 mm，N=0.0 kg/ha，NLCM=0.0 kg/ha。
- seed2 严格成功：False。
- 三个 seed 中按同一固定选点规则严格成功的 seed 数：1/3。
- 500-step smoke 正常结束；其结果仅用于检查流程，不用于策略判断。
- 所有正式 checkpoint 的动作与 MgmtEvent 总量最大差异：灌溉 0.000 mm，施氮 0.000 kg/ha。
- 正式 seed2 与 seed0 的 4 个输入文件 SHA-256 全部一致。
- 结论只针对同一 HLA2010 训练年和当前固定框架，不外推为跨站点或跨年份稳定。
