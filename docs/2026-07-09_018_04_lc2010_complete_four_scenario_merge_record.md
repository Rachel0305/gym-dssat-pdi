# 018_04 LC2010 完整四情景补齐记录

## 做了什么

018_03 中 LC2010 的 null / recorded / DSSAT auto 曾只使用旧脚本常量，缺少生物量、资源投入和胁迫字段。本轮不训练、不重跑 DSSAT，只从已经存在的 LC 017_11 完整筛选结果中抽取 LC2010 三个传统情景，并合并 LC DQN seed0 最佳奖励 checkpoint 与 018_03 官方推广 expert。

## LC2010 完整对照表

| site | station | year | scenario_label | grain_yield_kg_ha | biomass_kg_ha | irrigation_mm | nitrogen_kg_ha | max_water_stress | max_nitrogen_stress | source_file | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LC | Luancheng | 2010 | Null | 8051.000 | 15541.000 | 0.000 | 0.000 | 0.504 | 0.019 | DSSAT_auto_validation\lc_fixed_input_year_screening_017_11\017_11_lc_fixed_input_summary.csv | LC 017_11 fixed-input screening complete baseline |
| LC | Luancheng | 2010 | Recorded/farmer practice | 8732.000 | 16324.000 | 130.000 | 250.000 | 0.000 | 0.019 | DSSAT_auto_validation\lc_fixed_input_year_screening_017_11\017_11_lc_fixed_input_summary.csv | LC 017_11 fixed-input screening complete baseline |
| LC | Luancheng | 2010 | DSSAT auto | 8738.000 | 16373.000 | 138.500 | 0.000 | 0.000 | 0.019 | DSSAT_auto_validation\lc_fixed_input_year_screening_017_11\017_11_lc_fixed_input_summary.csv | LC 017_11 fixed-input screening complete baseline |
| LC | Luancheng | 2010 | DQN best checkpoint | 8739.000 | 16377.000 | 90.000 | 0.000 | 0.000 | 0.019 | DSSAT_auto_validation\lc2010_baseline_relative_dqn_smoke_017_12\seed0_5000steps\checkpoint_summary.csv | LC DQN seed0 best reward checkpoint=5000 |
| LC | Luancheng | 2010 | Official extension expert fixed DAP | 8739.000 | 16376.000 | 198.800 | 247.000 | 0.000 | 0.019 | DSSAT_auto_validation\extension_expert_baseline_018_03\018_03_extension_expert_summary.csv | official extension schedule fixed DAP |

## 输出

- LC complete: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_04_lc2010_complete_comparison.csv`
- Updated multisite clean comparison: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_03_clean_multisite_comparison_with_extension_expert.csv`

## 结论

LC2010 现在已经和其他站点一样具备完整字段。当前 LC2010 的 DQN seed0 最佳奖励 checkpoint 产量为 8739 kg/ha，灌溉 90 mm，施氮 0 kg/ha；与 DSSAT auto 8738 kg/ha 接近并略高，且比 recorded/farmer practice 8732 kg/ha 略高。官方推广 expert 产量 8739 kg/ha，但投入约 198.8 mm 灌溉和 247 kg/ha 氮。因此 LC2010 在 seed0 上也支持“低投入达到高产平台”的叙事，但此前 seed1 显示资源使用不稳定，不能直接作为跨 seed 稳定成功结论。
