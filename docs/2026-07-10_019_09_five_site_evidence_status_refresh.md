# 019_09 五站点已有证据状态刷新

## 结论先行

本轮没有重新审计或重新训练，而是刷新 019_01 已有证据矩阵。五站都已有代表年份和 DQN 证据，但目前不能说五站都已稳定满足导师目标。

当前最重要的新口径是：现有证据能直接支持产量、灌溉总量和施氮总量比较，但尚未形成统一 WUE/NUE 指标，因此不能把投入量占优自动表述为利用效率已经占优。
在正式 WUE/NUE 口径确定前，本表用 Pareto 判据做初筛：DQN 产量不低于基线（仅允许 1 kg/ha 数值容差），且灌溉量和施氮量均不高于基线。该判据评价的是产量-投入组合，不等同于正式 WUE/NUE。

## 当前五站决策表

| site | station | representative_year | optimization_space | dqn_yield | dqn_irrigation | dqn_nitrogen | yield_diff_vs_auto | yield_diff_vs_extension | input_dominates_auto | input_dominates_extension | pareto_dominates_auto_1kg_tolerance | pareto_dominates_extension_1kg_tolerance | seed_status | n_cost_offline_result | ic_status | evidence_classification | wue_nue_claim | leaching_reward | baseline_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | Hailun | 2010 | 明确 | 7853.665 | 120 | 0 | -0.335 | -0.335 | True | True | True | True | resource_stable_but_yield_not_stable | n_cost_no_selection_change | IC=1 主线；不因结果继续修改 | 资源效率候选；跨seed产量不稳定 | 未统一计算；当前只能声称产量/投入量差异 | 技术链路已通；暂不纳入正式reward | DSSAT_auto_validation/HLA_2004/hla_2010_2015_final_dqn_four_scenario_015_16/hla_2010_four_scenario_final_dqn_seed0_summary.csv |
| YC | Yucheng | 2014 | 明确 | 9418 | 120 | 250 | 705 | 1 | False | False | False | False | stable_yield_but_nitrogen_not_stable | n_cost_no_selection_change | 当前输入可用；不优先修改 | 产量跨seed稳定；施氮量不稳定 | 未统一计算；当前只能声称产量/投入量差异 | 技术链路已通；暂不纳入正式reward | DSSAT_auto_validation/yc2014_formal_four_scenario_015_06/seed0_seed1_best/015_06_yc2014_formal_four_scenario_summary.csv |
| FQ | Fengqiu | 2016 | 局部且年份敏感 | 7995 | 60 | 0 | -17 | 55 | False | True | False | True | seed1_success_but_seed0_not_reproduced | n_cost_no_selection_change | 年份敏感；禁止为提高胜率随意修改 | 单seed近平台且节氮；尚未跨seed复现 | 未统一计算；当前只能声称产量/投入量差异 | 技术链路已通；暂不纳入正式reward | DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv |
| SY | Shenyang | 2014 | 明确但资源投入高 | 11216 | 120 | 300 | 8492 | 139 | False | True | False | True | stable_high_yield_across_seed_but_high_resource | n_cost_no_selection_change | 2014 使用经诊断的 IC=2；必须保留敏感性说明 | 跨seed高产复现；未对auto实现投入量占优 | 未统一计算；当前只能声称产量/投入量差异 | 技术链路已通；暂不纳入正式reward | DSSAT_auto_validation/sy2014_dqn_resource_space_017_09/017_09_sy2014_four_scenario_summary.csv |
| LC | Luancheng | 2010 | 有限但真实 | 8739 | 90 | 0 | 1 | 0 | True | True | True | True | yield_stable_resource_unstable | n_cost_no_selection_change | 土壤ID/日期链路已修；保留修复记录 | 产量跨seed稳定；资源投入不稳定 | 未统一计算；当前只能声称产量/投入量差异 | 技术链路已通；暂不纳入正式reward | DSSAT_auto_validation\lc2010_baseline_relative_dqn_smoke_017_12\seed0_5000steps\checkpoint_summary.csv |

## 证据质量与剩余缺口

| issue | severity | evidence | impact | minimum_next_action |
| --- | --- | --- | --- | --- |
| WUE/NUE定义缺失 | high | 现有总表比较产量、I和N总量，未统一计算WUE/NUE；auto常出现N=0 | 不能把投入量较低直接写成水氮利用效率已超过 | 先与导师确定WUE/NUE公式、分母和N=0处理规则，再离线计算 |
| 跨seed稳定性不完整 | high | HLA产量不稳定；YC氮用量不稳定；FQ仅seed1成功；LC资源用量不稳定 | 除SY高产复现外，多数站点不能称为稳定成功 | 仅对最终候选框架补最小seed，不重复旧训练 |
| SY自动管理弱基线 | medium | SY2014 auto产量显著低于recorded/extension，且N=0 | 超过auto不能单独证明DQN优越；应重点比较官方推广expert | 汇报时分别列auto和extension，不合并成单一专家结论 |
| N-cost离线重评分无选择变化 | medium | 019_02中五站从N cost 5提高到20均未改变现有checkpoint选择 | 不能宣称单纯调高N成本已解决高N策略 | 保持正式reward；若导师要求再做统一小规模重训练，而非站点单独调参 |
| 淋洗惩罚训练不稳定 | medium | 019_07中FQ2016 leaching_cost=20在2000步较好，但4000/5000步退化 | 淋洗项目前不适合作正式reward组成 | 保留为环境效益敏感性；正式主线暂不加入 |

## 数据源完整性

| source | exists | rows | columns | size_bytes |
| --- | --- | --- | --- | --- |
| DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv | True | 25 | 13 | 5783 |
| DSSAT_auto_validation/extension_expert_baseline_018_03/018_10_hla_yc_fq_seed_recheck/018_10_site_recheck_status.csv | True | 3 | 13 | 976 |
| DSSAT_auto_validation/extension_expert_baseline_018_03/018_06_lc2010_seed_stability_audit/018_06_lc2010_seed_best_summary.csv | True | 4 | 15 | 752 |
| DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/018_08_seed0_vs_seed1_comparison.csv | True | 2 | 8 | 239 |
| DSSAT_auto_validation/reward_sensitivity_019_02/019_02_site_level_interpretation.csv | True | 5 | 8 | 1068 |
| docs/2026-07-10_019_08_keep_leaching_as_sensitivity_not_main_reward.md | True | n/a | n/a | 2017 |

## 数据质量检查

| check | result | value | severity_if_failed |
| --- | --- | --- | --- |
| 五站每站五情景 | pass | {'FQ': 5, 'HLA': 5, 'LC': 5, 'SY': 5, 'YC': 5} | critical |
| 站点-情景键唯一 | pass | 0 | critical |
| 产量/灌溉/施氮非负 | pass | 0 | high |
| 每站存在DQN情景 | pass | True | critical |
| 仅文件名的source_file | warn | 4 | low |

## 已关闭的重复工作

- 不再重做五站优化空间审计；019_01 已完成。
- 不再重复 gym/PDI 与 DSSAT 传输核查；旧链路证据继续有效。
- 不再通过随意降低 IC 制造优化空间。
- 不继续把淋洗惩罚并入正式 reward；019_08 已决定保留为敏感性扩展。
- 不把离线 N-cost 重评分解释成新策略已经学会节氮。

## 下一步最小工作

1. 先确定论文中 WUE/NUE 的正式定义和 N=0 情况的处理方式，并利用现有数据离线计算，不训练。
2. 将五站按证据等级分组：HLA/YC/FQ/LC 保留各自的稳定性限制；SY 是跨 seed 高产复现，但不是对 auto 的投入量双占优。
3. 只有在指标口径确定后，才决定哪一个站点需要补最小 seed 或统一 reward 小试。
