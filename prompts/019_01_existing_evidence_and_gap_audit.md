# 019_01 五站点已有证据与缺口审计

## 目的

用户提醒：此前已经做过大量站点和年份的优化空间审计，不能从零开始重复跑。

本轮任务只做低成本整理：

- 盘点五站点已有证据；
- 判断哪些问题已经回答；
- 标出仍缺的证据；
- 给出下一步实验优先级；
- 不新增 DSSAT 模拟；
- 不新增 DQN 训练。

## 核心问题

围绕导师确认的目标：

> DQN 策略尽量同时超过 expert 和 DSSAT auto 的产量与水氮利用效率。

需要把已有证据整理到以下问题下：

1. 五个站点是否都有继续优化潜力？
2. 哪些站点年份已经接近或达到导师目标？
3. 哪些站点只是高产但不节水/节氮？
4. 哪些站点需要先修输入或初始条件？
5. 是否已经有证据支持修改 reward 参数？
6. 是否已经有证据支持加入氮淋洗惩罚？

## 输入证据

优先使用这些已有记录和结果：

- `docs/2026-07-07_018_01_multisite_dqn_potential_reward_ic_audit.md`
- `docs/2026-07-09_018_10_hla_yc_fq_seed_recheck_record.md`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_09_multisite_status_refresh/`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_10_hla_yc_fq_seed_recheck/`
- `DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/`
- `DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/`
- `DSSAT_auto_validation/yc2014_cross_year_transfer_summary_report_016_12_fixed/`
- `DSSAT_auto_validation/fq2016_four_scenario_process_017_02/`
- `DSSAT_auto_validation/fq2019_process_plot_and_seed1_stability_017_06/`
- `DSSAT_auto_validation/HLA_2004/hla2010_to_2015_dqn_transfer_eval_015_17/`
- `DSSAT_auto_validation/HLA_2004/hla2010_to_2007_dqn_transfer_eval_015_18/`
- `DSSAT_auto_validation/HLA_2004/hla2010_to_2016_2022_dqn_transfer_eval_015_19/`

## 输出

- `DSSAT_auto_validation/five_site_strategy_design_019_01/019_01_existing_evidence_inventory.csv`
- `DSSAT_auto_validation/five_site_strategy_design_019_01/019_01_current_site_status_matrix.csv`
- `DSSAT_auto_validation/five_site_strategy_design_019_01/019_01_next_experiment_decision_table.csv`
- `docs/2026-07-09_019_01_existing_evidence_gap_audit_record.md`

## 原则

- 只整理，不重跑；
- 不把 promising 单 seed 说成稳定成功；
- 不把“高产”偷换成“节水节氮成功”；
- reward 和 IC 修改只能作为后续明确实验，不混入已有主线结果。
