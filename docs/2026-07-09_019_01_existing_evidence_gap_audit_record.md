# 019_01 五站点已有证据与缺口审计

## 结论先行

此前已经完成了大量优化空间和DQN结果审计，本轮不需要从零重跑。
当前最重要的工作是把已有证据整理成决策矩阵，再针对缺口做小实验。

## 已有证据索引

| topic | path | role | exists |
| --- | --- | --- | --- |
| 五站点DQN潜力/IC/reward总审计 | docs/2026-07-07_018_01_multisite_dqn_potential_reward_ic_audit.md | 已有最高层综述，回答五站点大方向 | yes |
| HLA/YC/FQ seed复核 | DSSAT_auto_validation/extension_expert_baseline_018_03/018_10_hla_yc_fq_seed_recheck/018_10_site_recheck_status.csv | 明确HLA/YC/FQ跨seed稳定性状态 | yes |
| SY2014 seed1复现 | DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/018_08_seed0_vs_seed1_comparison.csv | 证明SY2014已有seed0/seed1复现证据 | yes |
| LC2010 seed稳定性 | DSSAT_auto_validation/extension_expert_baseline_018_03/018_06_lc2010_seed_stability_audit/018_06_lc2010_seed_best_summary.csv | 证明LC产量稳定但节氮不稳定 | yes |
| official extension expert基线 | DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv | 导师认可的农技推广方案基线 | yes |
| YC跨年迁移 | DSSAT_auto_validation/yc2014_cross_year_transfer_summary_report_016_12_fixed/yc_cross_year_transfer_summary_table.csv | YC训练年到其他年份迁移表现 | yes |
| FQ2016四情景 | DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv | FQ2016代表性四情景结果 | yes |
| FQ2019过程图/seed | DSSAT_auto_validation/fq2019_process_plot_and_seed1_stability_017_06/fq2019_seed0_ckpt10000_four_scenario_summary.csv | FQ2019高产但高N案例 | yes |
| LC年份筛选 | DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/017_11_lc_fixed_input_summary.csv | LC哪些年份有优化空间 | yes |
| HLA跨年迁移汇总 | DSSAT_auto_validation/HLA_2004/hla2010_to_2016_2022_dqn_transfer_eval_015_19/hla2010_to_2016_2022_transfer_eval_summary.csv | HLA2010模型迁移到2016/2022 | yes |

## 当前五站点状态矩阵

| site | representative_years | optimization_space | best_current_role | current_problem | ic_status | reward_status | leaching_priority | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | 2010, 2015, 2007, 2016, 2022 | yes | 同站点跨年份迁移正例 | HLA2010本地seed1产量低于seed0，局部seed稳定性不足 | IC=1后更合理，已作为主线基础之一 | 当前baseline-relative reward可用；最佳策略多为N=0，暂不急着加N惩罚 | low_to_medium | 保留为迁移成功主线，同时诊断seed1为什么低产；不建议先改IC |
| YC | 2014 plus transfer years 2006/2009/2015/2018 | yes | 产量稳定正例，但主要靠较高施氮 | seed0/seed1产量稳定，但N用量250-300，节氮方向不稳定 | 暂不优先改IC | 最需要做N cost或leaching敏感性 | high | 做统一reward敏感性：N cost 5/10/20，必要时加氮淋洗项 |
| FQ | 2016, 2019 | partial | 年份敏感案例 | FQ2016成功主要来自seed1；seed0未复现，FQ2019高产但打满N300 | 不建议为追求结果随意改IC | 需要先区分低优化空间年份和高N成功年份 | high | 先筛选FQ可优化年份，再对2019类高N结果做N cost/leaching诊断 |
| SY | 2014 | yes | 跨seed高产复现案例 | DQN高产但I120/N300资源用量高，不是节水节氮成功 | SY曾有IC敏感问题，2014用IC=2修复过 | 需要抑制高N/高水策略，适合reward参数或leaching诊断 | high | 不要继续长训练；做资源效率导向reward敏感性 |
| LC | 2010 | limited_but_real | 输入修复后的小优化空间案例 | seed0节水节氮，seed1打满N300；资源效率不稳定 | 已修土壤ID/SDATE等输入链路，需保留记录 | 需要诊断seed1为何打满N | medium | 先诊断seed1动作/奖励，不急着改IC或长训练 |

## 下一步实验决策表

| priority | experiment | purpose | compute_cost | decision_rule |
| --- | --- | --- | --- | --- |
| 1 | 五站点证据矩阵归档 | 把已有审计、seed、IC、official expert结果统一成可汇报口径 | none | 完成后不再重复旧优化空间审计 |
| 2 | reward参数统一敏感性小试 | 检查N cost升高是否能减少YC/FQ/SY/LC打满N问题 | low_to_medium | 若N cost 10或20可保产降N，再进入seed复核 |
| 3 | 氮淋洗变量可用性审计 | 确认PDI/gym输出是否有稳定可用的leaching变量 | low | 变量可靠且高N情景显著增加淋洗时，才加入reward |
| 4 | IC敏感性限定诊断 | 只对输入异常或IC敏感站点做，不作为提高DQN胜率手段 | low | 只有能改善DSSAT农学合理性时才进入主线 |
| 5 | DQN正式训练/迁移扩展 | 在reward/IC口径明确后，对选定站点年份做seed0/seed1和跨年迁移 | medium_to_high | 只对有明确优化空间且reward口径稳定的站点执行 |

## 当前建议

1. 不重复已有优化空间审计。
2. 不马上修改初始条件，除非是输入异常或农学合理性问题。
3. 下一步优先做统一reward参数敏感性，尤其是N cost和氮淋洗变量可用性。
4. DQN长训练只放在reward和IC口径明确后的候选站点年份上。
