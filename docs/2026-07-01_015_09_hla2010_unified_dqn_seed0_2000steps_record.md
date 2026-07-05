# 015_09 HLA2010 统一 DQN 长训练记录

## 设置

- 年份：HLA2010
- 算法：DQN, seed=0
- 总步数：2000
- checkpoint 间隔：1000
- 奖励：max(0, delta_grnwt) - 1.0 * irrigation - 5.0 * nitrogen
- 动作空间：9-action，I in {0,15,30} mm, N in {0,50,100} kg/ha
- 预算：I<=120 mm, N<=300 kg/ha
- 单次上限：I<=30 mm, N<=100 kg/ha
- 最小间隔：7 days
- 管理模式：IRRIG=L, FERTI=L

## 输出文件

- 日值：`DSSAT_auto_validation/HLA_2004/hla_unified_dqn_long_train_015_09/2010/smoke_seed0_2000steps/dqn_eval_daily.csv`
- 汇总：`DSSAT_auto_validation/HLA_2004/hla_unified_dqn_long_train_015_09/2010/smoke_seed0_2000steps/checkpoint_summary.csv`
- 过程图：`DSSAT_auto_validation/HLA_2004/hla_unified_dqn_long_train_015_09/2010/smoke_seed0_2000steps/figures/hla2010_unified_dqn_seed0_2000steps_checkpoint_diagnostic.png`

## Checkpoint 结果

| checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 120 | 300 | 7854 | 20886 | 0.424 | 0.016 | 6233.665 |
| 2000 | 120 | 300 | 7854 | 20877 | 0.416 | 0.016 | 6233.665 |

## 关键判断

- 最高产量 checkpoint：1000，产量 7854.0 kg/ha，I=120.0 mm，N=300.0 kg/ha。
- 最高 reward checkpoint：1000，reward=6233.7，产量 7854.0 kg/ha。
- final checkpoint：2000，产量 7854.0 kg/ha，I=120.0 mm，N=300.0 kg/ha。
- 如果 best checkpoint 明显优于 final model，则正式比较应采用 checkpoint selection，而不是只看 final。