# 036_05：036_04代表PPO策略的措施合理性审计

## 任务目的

对 036_04 选中的每站点代表 checkpoint 进行“措施是否合理”的独立审计。该任务只读取 036_04 的选中年份和 036_01 的 daily CSV，不训练、不修改模型、不修改 reward。

## 重点问题

检查 PPO 的灌溉/施氮措施是否存在以下问题：

1. 早期集中投入：DAP1-10 是否使用了大部分水肥；
2. 频繁小灌：灌溉事件是否过多且单次剂量偏小；
3. 后期施氮：DAP90 后是否仍有施氮；
4. 最小间隔：同类水/氮事件之间是否小于 7 天；
5. 胁迫响应性：操作发生前 3 天是否已经存在水分或氮胁迫；
6. “最终指标成功”与“措施合理”是否一致。

## 输入

- `benchmark_results/036_04_select_checkpoint_and_plot_03601_03603_summary/tables/036_04_selected_year_level_comparison.csv`
- 其中记录的 `daily_csv_path`

## 输出

- `benchmark_results/036_05_selected_ppo_management_rationality_audit/tables/036_05_event_level_audit.csv`
- `benchmark_results/036_05_selected_ppo_management_rationality_audit/tables/036_05_year_level_management_audit.csv`
- `benchmark_results/036_05_selected_ppo_management_rationality_audit/tables/036_05_station_level_management_audit.csv`
- `benchmark_results/036_05_selected_ppo_management_rationality_audit/figures/036_05_ppo_event_timeline_by_station_year.png`
- `benchmark_results/036_05_selected_ppo_management_rationality_audit/figures/036_05_management_risk_counts_by_station.png`
- `docs/036_05_selected_ppo_management_rationality_audit_record.md`

## 判读原则

- 最终指标超过四情景最高值，只说明“结果上有候选价值”；
- 若出现大量早期集中投入、非胁迫响应、异常零产量或后期施氮，则必须标注为“措施合理性存疑”；
- 不因为最终指标好就自动认定措施合理；
- 不因为措施看起来怪就自动否定结果，应将其作为下一步反事实验证或 reward/约束修正的依据。

