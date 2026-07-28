# 038_02 GitHub 待提交候选清单

记录时间：2026-07-28

## 这份文件是干嘛的

这不是实验记录，而是准备提交 GitHub 前的“待提交区说明书”。

当前项目里未跟踪文件非常多，包含：

- 有用的 prompt；
- 有用的实验记录；
- 有用的代码；
- 有用的小图和汇总表；
- 也包含大量 runtime、中间 DSSAT 输入输出、旧错误结果、历史尝试、大体积目录。

因此本次不能使用：

```powershell
git add .
```

只能按清单逐批 `git add`。

## 当前已经确认可以提交的变更

### 1. 数据源切换导致的 my_data 模板删除

用户已确认以下删除是有意的：

```text
my_data/UFGA8201-FQ.jinja2
my_data/UFGA8201-HL.jinja2
my_data/UFGA8201-LC.jinja2
my_data/UFGA8201-SY.jinja2
my_data/UFGA8201-YC.jinja2
```

原因：训练数据源已切换为：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/
```

提交说明中必须写明这一点。

### 2. 新增说明文档

建议提交：

```text
docs/038_00_workspace_cleanup_plan.md
docs/038_01_current_mainline_index.md
docs/038_02_github_stage_candidate_list.md
```

## 建议第一批提交的内容

第一批提交只做“整理和导航”，不提交大结果、不提交模型、不提交 runtime。

建议命令：

```powershell
git add docs/038_00_workspace_cleanup_plan.md
git add docs/038_01_current_mainline_index.md
git add docs/038_02_github_stage_candidate_list.md
git add my_data/UFGA8201-FQ.jinja2
git add my_data/UFGA8201-HL.jinja2
git add my_data/UFGA8201-LC.jinja2
git add my_data/UFGA8201-SY.jinja2
git add my_data/UFGA8201-YC.jinja2
```

检查：

```powershell
git diff --cached --name-status
```

预期应该看到：

```text
A docs/038_00_workspace_cleanup_plan.md
A docs/038_01_current_mainline_index.md
A docs/038_02_github_stage_candidate_list.md
D my_data/UFGA8201-FQ.jinja2
D my_data/UFGA8201-HL.jinja2
D my_data/UFGA8201-LC.jinja2
D my_data/UFGA8201-SY.jinja2
D my_data/UFGA8201-YC.jinja2
```

## 建议第二批提交的内容

第二批提交建议只提交“当前主线最相关的 prompt、docs、src”，不提交 benchmark 大目录。

建议优先提交：

```text
prompts/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness.md
prompts/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun.md
prompts/036_02_correct_03601_baseline_metric_postprocess.md
prompts/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018.md
prompts/036_04_select_checkpoint_and_plot_03601_03603_summary.md
prompts/036_05_selected_ppo_management_rationality_audit.md
prompts/036_06_one_site_year_five_scenario_process_plot.md
prompts/036_07_five_station_representative_five_scenario_process_plots.md

docs/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness_record.md
docs/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun_record.md
docs/036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_summary.md
docs/036_02_correct_03601_baseline_metric_postprocess_record.md
docs/036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018_record.md
docs/036_04_select_checkpoint_and_plot_03601_03603_summary_record.md
docs/036_05_selected_ppo_management_rationality_audit_record.md
docs/036_06_one_site_year_five_scenario_process_plot_record.md
docs/036_07_five_station_representative_five_scenario_process_plots_record.md

src/readiness_original_free_timing_maskableppo_ic1_linked_036_00.py
src/run_original_free_timing_maskableppo_ic1_linked_five_site_half_split_036_01.py
src/postprocess_03601_baseline_metric_036_02.py
src/replay_03601_checkpoints_for_wp_et_036_03.py
src/select_checkpoint_and_plot_03601_03603_summary_036_04.py
src/audit_selected_ppo_management_rationality_036_05.py
src/build_one_site_year_five_scenario_process_plot_036_06.py
src/build_five_station_representative_process_plots_036_07.py
```

这些内容代表 IC=1、linked input 后的五站点 PPO 重跑主线。

## 建议第三批提交的内容

第三批提交建议提交 037 系列管理链与可信基线修复：

```text
prompts/037_00_036_results_metric_and_management_audit.md
prompts/037_01_036_management_problem_type_audit.md
prompts/037_02_lca_early_starter_cap_maskableppo_smoke.md
prompts/037_03_lca_early_starter_cap_maskableppo_smoke_clean_retrain.md
prompts/037_04_lc2019_early_starter_cap_50k_five_scenario_daily.md
prompts/037_05_dssat_management_event_chain_preflight.md
prompts/037_06_lc2019_linked_four_baseline_smoke.md
prompts/037_07_static_level1_four_baseline_rebuild.md
prompts/037_08_FQA_validation_ppo_fixed_baseline_figures.md

docs/037_00_036_results_metric_and_management_audit_record.md
docs/037_01_036_management_problem_type_audit_record.md
docs/037_02_invalid_cache_reuse_note.md
docs/037_02_lca_early_starter_cap_maskableppo_smoke_record.md
docs/037_03_lca_early_starter_cap_maskableppo_smoke_clean_retrain_record.md
docs/037_04_lc2019_baseline_warning.md
docs/037_04_lc2019_early_starter_cap_50k_five_scenario_daily_record.md
docs/037_05_dssat_management_event_chain_preflight_record.md
docs/037_06_lc2019_linked_four_baseline_smoke_record.md
docs/037_07_FQA_only_static_level1_baseline_result.md
docs/037_07_static_level1_four_baseline_rebuild_record.md
docs/037_08_FQA_validation_ppo_fixed_baseline_figures_record.md

src/audit_036_results_metric_and_management_037_00.py
src/audit_036_management_problem_type_037_01.py
src/run_lca_early_starter_cap_maskableppo_smoke_037_02.py
src/run_lca_early_starter_cap_maskableppo_smoke_037_03.py
src/build_lc2019_03703_50k_five_scenario_daily_037_04.py
src/audit_dssat_management_event_chain_037_05.py
src/run_lc2019_linked_four_baseline_smoke_037_06.py
src/run_static_level1_four_baseline_rebuild_037_07.py
src/build_fqa_validation_ppo_fixed_baseline_figures_037_08.py
```

这些内容代表“发现旧结果不可信、修管理链、重建可信基线”的关键证据链。

## 暂不建议提交的内容

### 1. benchmark_results 大目录整体

暂不建议提交整个：

```text
benchmark_results/
```

原因：

- 体积很大；
- 包含 runtime、snapshot、rendered input、失败尝试；
- 部分结果是历史错误口径；
- GitHub 不适合保存大量 DSSAT 中间输出和模型文件。

如果需要提交图表，建议只挑选小体积、最终可信的代表性图和 summary 表。

### 2. proposal_materials 修改

当前有一份开题答辩讲稿 markdown 被修改。它和本次 RL 清理不是同一件事，暂不纳入提交，除非用户明确要求。

### 3. 031、032、035 系列

这些系列有历史诊断价值，但不是当前最终主线。建议先不提交，等 036/037 主线备份完成后，再决定是否作为“方法演进记录”分批提交。

## 推荐提交顺序

推荐至少分三次 commit：

1. `docs: add workspace cleanup and mainline index`
2. `exp036: add IC1 linked PPO rerun prompts records and scripts`
3. `exp037: add management-chain audit and trusted baseline rebuild`

这样以后 Git 历史会清楚很多，不会一个 commit 塞进几百个文件。

