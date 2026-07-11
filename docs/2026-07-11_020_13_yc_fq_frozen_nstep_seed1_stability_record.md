# 020_13 YC/FQ 冻结 n-step DQN seed1 稳定性复核记录

## 设计

- YC2014、FQ2016均在020_12冻结框架下独立训练seed1 50K；相对seed0只更换模型随机种子。
- 奖励、9动作、I120/N300、7 DAP间隔、n_steps=5、目标站点输入和本地null全部保持不变。
- 每5K确定性评估，按total reward最大、并列取最早checkpoint。

## DQN跨seed结果

| site | dqn_seed | checkpoint_step | final_grain_kg_ha | irrigation_mm | nitrogen_kg_ha | unified_reward | yield_diff_vs_auto_kg_ha | yield_diff_vs_extension_expert_kg_ha | yield_range_within_site_kg_ha | irrigation_range_within_site_mm | nitrogen_range_within_site_kg_ha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | 0.000 | 50000.000 | 7985.000 | 120.000 | 50.000 | 548.564 | -27.000 | 45.000 | 0.000 | 15.000 | 50.000 |
| FQ | 1.000 | 40000.000 | 7985.000 | 105.000 | 0.000 | 813.536 | -27.000 | 45.000 | 0.000 | 15.000 | 50.000 |
| YC | 0.000 | 35000.000 | 8659.000 | 60.000 | 0.000 | 774.153 | -54.000 | -758.000 | 17.000 | 30.000 | 0.000 |
| YC | 1.000 | 50000.000 | 8676.000 | 90.000 | 0.000 | 760.573 | -37.000 | -741.000 | 17.000 | 30.000 | 0.000 |

## 运行审计

| site | year | seed | checkpoint_count | all_runtime_audits_passed | source_file |
| --- | --- | --- | --- | --- | --- |
| YC | 2014 | 0 | 10 | 1 | DSSAT_auto_validation/frozen_nstep_cross_site_020_12/YC2014/seed0_50000steps/reaudit_operation_dap_020_12/corrected_runtime_audit_summary.csv |
| YC | 2014 | 1 | 10 | 1 | DSSAT_auto_validation/frozen_nstep_cross_site_020_12/YC2014/seed1_50000steps/runtime_audit_summary.csv |
| FQ | 2016 | 0 | 10 | 1 | DSSAT_auto_validation/frozen_nstep_cross_site_020_12/FQ2016/seed0_50000steps/runtime_audit_summary.csv |
| FQ | 2016 | 1 | 10 | 1 | DSSAT_auto_validation/frozen_nstep_cross_site_020_12/FQ2016/seed1_50000steps/runtime_audit_summary.csv |

## 结论

- YC2014：seed0/seed1产量为8659/8676 kg/ha，差17 kg/ha；两者均N0，灌溉为60/90 mm。产量和不施氮方向可复现，但精确灌溉量尚不稳定。两者都接近但低于auto 8713，也明显低于recorded和官方推广expert约9417–9418。
- FQ2016：两个seed产量均为7985 kg/ha；灌溉为120/105 mm，施氮为50/0 kg/ha。产量高度稳定、用水较接近，但是否施氮尚未稳定。两者均低于auto 8012，且用水高于auto 59.9 mm。
- 因此可以说同一冻结框架在YC/FQ上跨seed复现了‘显著优于null、接近强基线’的产量水平；不能说已跨seed复现了完全相同的水氮动作，也不能说全面超过auto和官方expert。
- 下一步若继续，应先由导师决定是否把‘接近auto且更省部分资源’作为可接受目标；在此之前不建议直接追加seed2或修改奖励。