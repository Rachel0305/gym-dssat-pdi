# 038_01 当前主线索引：哪些结果可信、哪些只是历史诊断

记录时间：2026-07-28

## 目的

这个文件不是新的实验结果，而是“导航图”。它回答三个问题：

1. 现在项目主线到底走到哪里？
2. 哪些代码、prompt、实验记录、结果可以继续引用？
3. 哪些旧结果只能作为历史诊断，不能再当作当前可信结果汇报？

## 当前数据源口径

当前训练与验证数据源应使用：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/
```

`my_data/UFGA8201-*.jinja2` 已由用户确认有意删除，因为数据源已经切换。后续 Git 提交时可以提交这些 tracked deletion，但提交说明必须写清楚“数据源迁移”，避免被误判为误删。

## 当前最重要的事实边界

### 1. 旧结果中存在两个关键问题

此前一批自由 PPO / DQN / 五情景图结果中，后来发现两个会影响可信度的问题：

- 初始条件 IC 相关设置曾经不符合当前预期；
- DSSAT 管理事件链中，模板管理模式/事件写入链条曾经需要修正和复核。

因此，旧结果不能简单延续为“最终可信结果”。它们仍然有诊断价值，但如果要汇报当前模型效果，必须优先使用后续在 IC 与 DSSAT 管理链修正后重建的结果。

### 2. 032 系列：自由时序与压力奖励探索线

032 系列主要价值是探索：

- 自由时序 PPO/DQN 能否运行；
- stress-aware reward 是否改善管理合理性；
- LC 多年训练/迁移是否可行；
- 五站点 half-split 批处理是否能跑通；
- 日过程图、累积奖励图、压力指数图如何生成。

但 032 系列是在后续 IC/管理链问题完全收束前产生的，因此：

- 可以作为算法探索、绘图模板、脚本来源；
- 不建议作为最终可信的数值结果直接汇报；
- 若引用，必须标注为“历史探索/诊断结果”。

关键文件：

```text
prompts/032_22_five_site_half_split_stress_aware_maskableppo_batch.md
docs/032_22_five_site_half_split_stress_aware_maskableppo_batch_record.md
src/run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py
benchmark_results/032_22_five_site_half_split_stress_aware_maskableppo_batch/
```

以及 LC 分站探索：

```text
docs/032_10_lc_multiyear_free_timing_ppo_smoke_record.md
docs/032_11_lc_multiyear_free_timing_ppo_training_length_record.md
docs/032_17_lc_75k_ppo_all_year_five_scenario_daily_package_record.md
docs/032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild_record.md
```

### 3. 036 系列：IC=1、linked input 后的五站点 half-split PPO 重跑线

036 系列是更接近当前主线的结果。它重新跑了五站点 half-split 的自由时序 MaskablePPO，并做了指标后处理、checkpoint 选择、管理合理性审计和代表性五情景图。

关键文件：

```text
prompts/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness.md
docs/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness_record.md
src/readiness_original_free_timing_maskableppo_ic1_linked_036_00.py

prompts/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun.md
docs/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun_record.md
docs/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_summary.md
src/run_original_free_timing_maskableppo_ic1_linked_five_site_half_split_036_01.py
benchmark_results/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun/
```

后处理与绘图：

```text
docs/036_02_correct_03601_baseline_metric_postprocess_record.md
src/postprocess_03601_baseline_metric_036_02.py

docs/036_04_select_checkpoint_and_plot_03601_03603_summary_record.md
src/select_checkpoint_and_plot_03601_03603_summary_036_04.py

docs/036_05_selected_ppo_management_rationality_audit_record.md
src/audit_selected_ppo_management_rationality_036_05.py

docs/036_07_five_station_representative_five_scenario_process_plots_record.md
src/build_five_station_representative_process_plots_036_07.py
benchmark_results/036_07_five_station_representative_five_scenario_process_plots/
```

036 系列可以作为“当前主线尝试”的重要材料，但需要结合 037 系列对管理链和基线可信度的修正来看。

### 4. 037 系列：管理合理性问题、DSSAT 事件链、可信基线修复线

037 系列是当前最需要优先保留和阅读的修正线。它的核心价值是发现并确认：

- 036 结果中存在管理措施合理性问题；
- 某些五情景/基线图不能直接作为可信对照；
- 必须先确保 DSSAT 管理事件进入模拟链条；
- 需要重建静态基线，再比较 PPO。

关键审计文件：

```text
docs/037_00_036_results_metric_and_management_audit_record.md
src/audit_036_results_metric_and_management_037_00.py

docs/037_01_036_management_problem_type_audit_record.md
src/audit_036_management_problem_type_037_01.py

docs/037_05_dssat_management_event_chain_preflight_record.md
src/audit_dssat_management_event_chain_037_05.py
benchmark_results/037_05_dssat_management_event_chain_preflight/
```

可信基线与 FQ 局部修复：

```text
docs/037_07_static_level1_four_baseline_rebuild_record.md
docs/037_07_FQA_only_static_level1_baseline_result.md
src/run_static_level1_four_baseline_rebuild_037_07.py
benchmark_results/037_07_static_level1_four_baseline_rebuild/

docs/037_08_FQA_validation_ppo_fixed_baseline_figures_record.md
src/build_fqa_validation_ppo_fixed_baseline_figures_037_08.py
benchmark_results/037_08_FQA_validation_ppo_fixed_baseline_figures/
```

当前如果要给导师展示“一个站点的可信结果”，应优先从 037_07/037_08 这条线取，而不是直接使用旧 036 图。

## 当前训练框架口径

当前主线讨论中的 PPO 框架是：

- 每日环境运行；
- agent 每日观察；
- 自由时序决策，不再固定 expert DAP；
- 使用 MaskablePPO；
- 动作空间包含灌溉档位和施氮档位；
- 包含单季水氮上限；
- 包含 DAP90 后禁氮；
- 是否包含最小操作间隔，需要以对应 prompt/代码为准；
- 训练数据源使用 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/`。

注意：因为近几轮出现过“我们以为约束保留，但某些运行口径可能不一致”的问题，后续任何新实验必须在 prompt 开头显式列出：

- 数据源；
- IC 是否启用；
- DSSAT 管理模式是否允许外部事件进入；
- 灌溉/施氮动作档位；
- 单季上限；
- 单次上限；
- 最小间隔；
- 后期禁氮；
- reward 公式；
- checkpoint 选择协议；
- 评价指标公式。

## 当前不建议直接引用为最终结果的内容

以下内容应保留，但不建议作为最终可信数值直接汇报：

- 021--031 系列 DQN/DQfD/阶段型 Q/监督排序调试结果；
- 032 系列中 IC/管理链完全修正前的五情景汇总结果；
- 036 系列中未经 037 管理链审计和静态基线修正的五情景图；
- `*_failed_attempt*`、`*_attempt*` 目录；
- 任何用旧 `my_data/` 或 IC=0 口径跑出的结果。

这些内容的价值是说明“为什么后来改成当前路线”，不是说明“当前模型已经达成什么最终效果”。

## GitHub 备份建议

建议提交：

- `prompts/032_*`、`prompts/036_*`、`prompts/037_*`
- `docs/032_*`、`docs/036_*`、`docs/037_*`
- 当前新增的 `docs/038_00_workspace_cleanup_plan.md`
- 当前新增的 `docs/038_01_current_mainline_index.md`
- `src/*032*`、`src/*036*`、`src/*037*`
- 小体积 summary CSV、指标表、代表性 figures

不建议提交：

- 模型 zip；
- DSSAT runtime；
- 大量中间 OUT；
- 重复 checkpoint；
- 临时缓存；
- 被判定为错误运行的大体积结果包。

## 后续任务建议

下一步不是继续清空文件，而是先建立一套“可信重跑”的最小主线：

1. 先统一确认输入数据、IC、DSSAT 管理事件链；
2. 先重建四基线；
3. 再训练/评估 PPO；
4. 再生成五情景图和汇总图；
5. 每一步都保存中文 prompt 和中文实验记录。

这和用户新的合作方式一致：Codex 先说明要改什么、为什么改、在哪个脚本里改、用户如何运行，而不是在用户不可见的情况下全权接管项目。

