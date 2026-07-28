# 037_00：036 主线结果指标表现与措施合理性总审计

## 任务背景

036_01 已经完成“正确 multisite_013 输入源 + IC=1 + DSSAT linked action + 自由时序 MaskablePPO + 原 stress-aware reward”的五站点 half-split 正式重跑。

036_04 已经选出每个站点的代表 checkpoint，并给出 50 个站点-年份的指标对照。

036_05 已经做过措施合理性审计，但记录文件存在编码显示问题，而且主要偏风险统计。本任务不重新训练、不重新评估 DSSAT，只读取 036_04 与 036_05 的既有结果，生成一份更适合汇报的中文总览。

## 输入文件

- `benchmark_results/036_04_select_checkpoint_and_plot_03601_03603_summary/tables/036_04_selected_year_level_comparison.csv`
- `benchmark_results/036_04_select_checkpoint_and_plot_03601_03603_summary/tables/036_04_selected_checkpoints.csv`
- `benchmark_results/036_05_selected_ppo_management_rationality_audit/tables/036_05_year_level_management_audit.csv`

## 审计口径

### 指标表现

同时保留两套口径：

1. 相对四情景最高值：
   - 产量 gap：`yield_gap_vs_four_max`
   - 水分生产力 gap：`wp_et_gap_vs_four_max`
   - 氮肥偏生产力 gap：`pfp_n_gap_vs_four_max`
   - 至少一个指标超过四情景最高值：`any_metric_win_four_max`

2. 相对 official expert：
   - 节水量：`water_saving_vs_expert_mm`
   - 节氮量：`n_saving_vs_expert_kg_ha`
   - 产量差：`yield_gap_vs_expert`
   - WP_ET 差：`wp_et_gap_vs_expert`
   - PFP_N 差：`pfp_n_gap_vs_expert`

这两套口径不能混用：四情景最高值用于回答“是否超过所有对照情景”，expert 口径用于回答“相对专家省了多少水氮”。

### 措施合理性

硬风险：

- `early_dump_warning`：DAP1-10 早期集中投入水或氮占季节总量 >= 80%；
- `interval_warning`：同类操作间隔 < 7 天；
- `late_n_warning`：DAP90 后仍施氮；
- `zero_yield_warning`：最终籽粒产量为 0。

解释性风险：

- `mostly_preventive_warning`：操作前 3 天没有明显水/氮胁迫，属于预防性操作。该项不直接判为错误，因为 expert 也包含预防性管理；但汇报时需要解释。

## 输出要求

生成：

- 年份级合并表；
- 站点级汇总表；
- 总体摘要；
- 指标胜出数量图；
- 措施风险数量图；
- 站点均值 gap 与节水节氮图；
- 中文实验记录 MD。

## 停止线

本任务只做后处理审计。不得训练模型，不得重跑 DSSAT，不得修改 reward、动作空间、约束条件或 checkpoint 选择规则。
