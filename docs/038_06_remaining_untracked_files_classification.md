# 038_06 剩余未跟踪文件分类表

记录时间：2026-07-28

## 当前状态概览

在已经提交并推送以下内容之后：

- 038 工作区清理和主线索引；
- 037 管理链审计与可信基线脚本；
- 032 自由时序 PPO 历史探索线；
- 031/035 reward 与管理诊断线；
- 037 FQ 可信小结果包；

工作区仍然有大量未跟踪文件。它们主要集中在：

- `benchmark_results/`
- `proposal_materials/`
- 少量 `docs/*.inspect.ndjson`

这些文件目前不建议继续无脑提交。

## 一、benchmark_results 剩余内容

### 1. 不建议上传的大类

以下类型不建议上传 GitHub：

- `runtime_train/`
- `runtime_eval/`
- `snapshot/`
- `snapshots/`
- `rendered_inputs/`
- `input/`
- DSSAT `.OUT`
- DSSAT 临时目录
- failed attempt 目录
- 大量旧中间结果目录

原因：

- 体积大；
- 可由脚本重建；
- 包含重复输入；
- 部分来自旧错误口径；
- 不适合作为 GitHub 主仓库存档。

### 2. 当前未跟踪 benchmark_results 主要分组

剩余未跟踪文件较多的目录包括：

```text
029_04_maskableppo_vs_maskaware_dqn_evidence
028_12_screened_year_representative_advisor_package
029_02_five_site_stage_mask_aware_dqn
027_07_site_specific_stage_maskable_ppo_attempt2
028_05_sy_crossyear_frozen_ppo_daily
028_04_existing_yc_fq_lc_ppo_frozen_daily
027_05
027_07_site_specific_stage_maskable_ppo
028_10_fq2016_missing_seed1_seed2
029_03_frozen_maskaware_dqn_crossyear
028_07_missing_official_expert_six_season_completion
031_35_missing_four_baseline_completion_for_03134
031_36_missing_dssat_auto_completion_for_03134
```

这些多数是历史结果包、早期 advisor package、DQN/PPO 比较、缺失基线补齐、跨年迁移结果等。它们有历史价值，但不建议整体提交。

### 3. 如果后续确实要备份

建议只从这些目录中挑：

- summary CSV；
- final metrics CSV；
- representative figures；
- manifest JSON；
- 关键 daily merged CSV；
- 少量 PPT 用图。

不建议提交完整目录。

推荐做法：

```text
每次只为一个主题建立一个 result artifact manifest，
说明挑了哪些文件、为什么挑、哪些没有上传。
```

## 二、proposal_materials 剩余内容

当前有：

```text
proposal_materials/...博士开题答辩讲稿-20分钟.md  # modified
proposal_materials/add_10_more_references_0717.py
proposal_materials/add_references_0717.py
proposal_materials/final_audit_render/
```

建议：

- 这些是开题/写作材料，不和当前 RL 实验结果混在一个 commit；
- 如果需要备份，应单独 commit，例如：

```text
proposal: backup defense script and reference tooling
```

当前不建议和 RL 结果一起 push。

## 三、docs inspect 文件

当前有：

```text
docs/2026-07-19_028_15_existing_results_advisor_presentation.pptx.inspect.ndjson
docs/2026-07-19_028_16_existing_results_advisor_presentation_plain_full_daily.pptx.inspect.ndjson
docs/2026-07-19_029_04_maskableppo_vs_maskaware_dqn_comparison.pptx.inspect.ndjson
```

这些是 PPT 检查中间产物。

建议：

- 暂不提交；
- 如果以后要复盘 PPT 制作过程，可单独提交；
- 如果只是临时 inspect 输出，可以保留本地，不进 GitHub。

## 四、建议后续整理顺序

### 优先级 1：当前主线新实验

下一步最重要的不是继续整理历史大结果，而是进入新实验前置：

```text
低初始土壤水分/氮条件派生数据审计与 smoke
```

建议新编号：

```text
039_00
```

### 优先级 2：选择性整理旧图

如果需要向导师展示旧 PPO/DQN 对比，可以单独整理：

```text
029_04_maskableppo_vs_maskaware_dqn_evidence
```

但只上传代表图、summary 和 manifest，不上传 runtime。

### 优先级 3：proposal 材料单独备份

如果要备份开题材料，单独 commit，不和 RL 实验混合。

## 当前建议

当前 GitHub 已经具备：

- 方法演进记录；
- 关键审计脚本；
- FQ 当前可信小结果包；
- 工作区导航说明。

继续整理历史大目录的边际收益已经不高。建议转入 039 新实验设计，而不是继续把旧 benchmark 大包塞进 GitHub。

