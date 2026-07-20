# 028_14 现有结果、官方推广专家管理与指标差距整理

日期：2026-07-19

## 目标

在不训练模型、不重跑 DSSAT、不修改 reward、IC、输入或既有结果的前提下，整理当前 17 个已筛选站点年份的既有证据，回答：

1. 官方推广 expert 在各生态区如何安排灌溉和施氮；
2. 当前阶段型 MaskablePPO 代表策略采取了什么措施；
3. 这些措施在预算、时机、胁迫和跨 seed 证据层面是否合理；
4. 相对 null、recorded、DSSAT auto、official expert 四个基线，产量、WP_ET、PFP_N 分别超过或落后多少；
5. 历史 DQN 结果目前处于什么证据等级。

## 数据源与口径

- 17 年五情景终值：`benchmark_results/028_12_screened_year_representative_advisor_package/028_12_all_representative_five_scenario_summary.csv`
- 17 年五情景日值：`benchmark_results/028_12_screened_year_representative_advisor_package/028_12_all_representative_five_scenario_daily.csv`
- PPO 精确终值与跨 seed：`benchmark_results/028_13_screened_year_maskableppo_advisor_summary/028_13_site_year_overview.csv`
- 官方专家原始阶段映射：`DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_extension_expert_schedule.csv`
- 历史 DQN 五站点代表证据：`benchmark_results/027_05/027_05_dqn_five_scenario_summary.csv`

指标定义：

- 产量：最终 grain yield，kg/ha；
- WP_ET：grain yield / seasonal ET，kg/m3；
- PFP_N：grain yield / applied N，kg grain/kg N；当 N=0 时保持 NA，不插补；
- “超过四基线”：代表 PPO 指标严格大于四个基线中的最大可比值；
- 差距：`PPO - four_baseline_max`，同时输出绝对值和百分比；
- 日值措施以实际 DSSAT 输出事件为准，不以计划表替代；专家 DAP100 等未在收获前执行的事件必须标记为季节截断。

## 合理性审计

逐站点年份检查：

- PPO 灌溉不超过 120 mm、施氮不超过 300 kg/ha；
- 所有动作位于预定义阶段；
- DAP>90 是否仍施氮；
- 最大水分/氮胁迫；
- PPO 与 official expert 的水氮投入差；
- winner seed 数和 seed 总数；
- 代表 seed 成功不能替代跨 seed 稳定性。

“措施合理”仅指预算合法、时点可解释、无明显晚期施氮且结果具有至少一项指标优势；不将该标签解释为真实田间最优或因果农艺证明。若跨 seed 少于 2/3，标记为“候选存在、稳定性不足”。

## 输出

- `benchmark_results/028_14_existing_results_report/028_14_expert_schedule_templates.csv`
- `benchmark_results/028_14_existing_results_report/028_14_expert_executed_events.csv`
- `benchmark_results/028_14_existing_results_report/028_14_ppo_metric_gap_detail.csv`
- `benchmark_results/028_14_existing_results_report/028_14_ppo_management_rationality.csv`
- `benchmark_results/028_14_existing_results_report/028_14_dqn_current_evidence.csv`
- `benchmark_results/028_14_existing_results_report/028_14_metric_gap_heatmap.png/.svg`
- `docs/2026-07-19_028_14_existing_results_expert_management_and_metric_gap_report.md`
- `docs/2026-07-19_028_14_existing_results_expert_management_and_metric_gap_report.html`
- `docs/2026-07-19_028_14_existing_results_expert_management_and_metric_gap_record.md`

## 硬约束

- 训练调用数必须为 0；DSSAT 新运行数必须为 0；
- 不覆盖 027/028 已有结果；
- 不把 historical DQN provisional 结果升级为正式结论；
- 不把单一代表 seed 冒充所有 seed；
- 不为 PFP_N 的 N=0 情景填造数值；
- 所有缺失、精度差异和早收获截断均需显式记录。
