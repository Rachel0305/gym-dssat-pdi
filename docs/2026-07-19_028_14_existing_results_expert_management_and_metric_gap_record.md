# 028_14 现有结果整理记录

## 状态

completed

## 执行范围

- 训练调用：0
- DSSAT新运行：0
- 输入、IC、reward、模型权重修改：0
- 复用17个站点年份、85个五情景终值和11155行日值。

## 数据质量检查

- 17个site-year均有5个情景；
- PPO精确终值来自028_13，日值来自028_12；
- expert实际措施从日值事件提取；
- PFP_N的N=0保持NA；
- 代表seed与跨seed证据分开；
- 产量、WP_ET、PFP_N分别有9、7、12个年份严格胜出；
- 17/17预算合法、阶段合法、DAP>90无施氮；
- 10/16三seed年份达到至少2/3初步稳定。

## 解释边界

合理性为模型约束与现有结果层面的审计，不是田间试验证明。official expert是推文区间中值的固定DAP映射。DQN保留provisional标签。

## 输出

- `benchmark_results\028_14_existing_results_report`
- `docs\2026-07-19_028_14_existing_results_expert_management_and_metric_gap_report.md`
- `docs\2026-07-19_028_14_existing_results_expert_management_and_metric_gap_report.html`
