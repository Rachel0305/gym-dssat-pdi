# 028_15 现有强化学习结果导师汇报PPT

## 目标

基于028_12、028_13和028_14已经冻结的结果，制作一套中文导师汇报PPT，说明官方推广expert的管理安排，并逐站点、逐年份展示五情景终值对照图和精确数据表。

## 边界

- 不训练模型；
- 不新增DSSAT运行；
- 不修改输入、IC、reward或模型；
- 不重算已冻结结果；
- PPO代表候选与跨seed稳定性必须分开；
- DQN历史证据保持provisional标签；
- PFP_N在N=0时保持NA。

## 数据源

- `benchmark_results/028_12_screened_year_representative_advisor_package/028_12_all_representative_five_scenario_summary.csv`
- `benchmark_results/028_12_screened_year_representative_advisor_package/<SITE>/<YEAR>/figures/*_five_scenario_endpoints.png`
- `benchmark_results/028_12_screened_year_representative_advisor_package/<SITE>/<YEAR>/figures/*_selected_stage_actions.png`
- `benchmark_results/028_14_existing_results_report/028_14_ppo_metric_gap_detail.csv`
- `benchmark_results/028_14_existing_results_report/028_14_ppo_management_rationality.csv`
- `benchmark_results/028_14_existing_results_report/028_14_expert_schedule_templates.csv`

## 幻灯片结构

1. 标题与研究问题；
2. 总体结论与三指标差距热图；
3. 判定口径与证据边界；
4. 两套官方推广expert阶段水肥模板；
5. 五站点结果总览；
6. 17个站点年份各两页：
   - 五情景终值对照图；
   - 五情景精确数据表、PPO阶段措施图、指标差距和seed证据；
7. 综合分级、DQN历史对照与建议。

## 验收

- 17/17站点年份均有五情景图和五行情景数据表；
- 85个五情景记录完整；
- 图表值与源CSV一致；
- 产量、WP_ET、PFP_N领先数量分别为9/17、7/17、12/17；
- 措施约束与10/16跨seed稳定结论如实呈现；
- 使用Artifact Tool生成PPTX；
- 渲染全部幻灯片，检查溢出、重叠、字体与图片清晰度；
- 记录所有失败和修复，不伪造完成。
