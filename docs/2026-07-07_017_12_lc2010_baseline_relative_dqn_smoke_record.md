# 017_12 LC2010 baseline-relative DQN smoke 记录

## 设置

- 站点年份：LC2010
- seed：1
- timesteps：5000
- checkpoint interval：1000
- 算法：Stable-Baselines3 DQN
- 奖励：max(0, GWAD_final - LC2010_null) - 1.0*I - 5.0*N
- null baseline：8051 kg/ha
- recorded expert：8732 kg/ha, I=130 mm, N=250 kg/ha
- DSSAT auto：8738 kg/ha, I=138.5 mm, N=0 kg/ha
- 动作空间：I in {0,15,30}, N in {0,50,100}
- 预算：I<=120 mm, N<=300 kg/ha, min interval=7 days

## 输出

- Run dir: `DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed1_5000steps`
- Summary: `DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed1_5000steps/checkpoint_summary.csv`
- Daily: `DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed1_5000steps/dqn_eval_daily.csv`
- Events: `DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed1_5000steps/dqn_eval_events.csv`

## checkpoint 结果

| checkpoint_step | final_grain_kg_ha | final_biomass_kg_ha | action_irrigation_total | action_fertilizer_total | event_irrigation_total | event_fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward | yield_diff_vs_recorded | yield_diff_vs_auto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000.000 | 8738.000 | 16371.000 | 120.000 | 300.000 | 120.000 | 300.000 | 0.000 | 0.019 | -932.993 | 6.000 | 0.000 |
| 2000.000 | 8738.000 | 16371.000 | 120.000 | 300.000 | 120.000 | 300.000 | 0.000 | 0.019 | -932.993 | 6.000 | 0.000 |
| 3000.000 | 8738.000 | 16372.000 | 120.000 | 300.000 | 120.000 | 300.000 | 0.000 | 0.019 | -932.726 | 6.000 | 0.000 |
| 4000.000 | 8738.000 | 16372.000 | 120.000 | 300.000 | 120.000 | 300.000 | 0.000 | 0.019 | -932.726 | 6.000 | 0.000 |
| 5000.000 | 8739.000 | 16374.000 | 120.000 | 300.000 | 120.000 | 300.000 | 0.000 | 0.019 | -932.467 | 7.000 | 1.000 |

## 初步判断

- 最高产量 checkpoint: 5000, GWAD=8739.0, I=120.0, N=300.0, reward=-932.5.
- 最高奖励 checkpoint: 5000, GWAD=8739.0, I=120.0, N=300.0, reward=-932.5.
- 本轮是 smoke，不作为跨 seed 稳定性结论；若出现候选 checkpoint，下一步才做 seed1。
