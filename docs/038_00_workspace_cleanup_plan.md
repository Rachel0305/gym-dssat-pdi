# 038_00 工作区清理计划与当前文件状态

记录时间：2026-07-28

## 一句话结论

当前工作区不能用 `git clean` 或批量删除命令一键清理，因为 dry-run 预览显示它会把 `DSSAT_auto_validation/` 下大量 DSSAT 输入、验证目录也列为可删除对象。后续清理必须采用“先分类、再确认、只清缓存、不动研究数据”的方式。

## 当前观察到的主要状态

### 1. Git 工作区状态

`git status --short` 显示：

- `my_data/UFGA8201-FQ.jinja2`
- `my_data/UFGA8201-HL.jinja2`
- `my_data/UFGA8201-LC.jinja2`
- `my_data/UFGA8201-SY.jinja2`
- `my_data/UFGA8201-YC.jinja2`

这 5 个模板文件处于 tracked deletion 状态。由于 `my_data/` 属于重要研究数据，不能由 Codex 直接删除或提交删除；需要用户确认这些删除是否是预期行为。

另有一份开题答辩讲稿 markdown 被修改：

- `proposal_materials/...博士开题答辩讲稿-20分钟.md`

这属于用户写作材料，暂不纳入本次 RL 结果清理，除非用户明确要求。

### 2. 大体积目录

按目录体积粗略统计，当前主要空间占用集中在：

- `benchmark_results/`：约 3.68 GB
- `DSSAT_auto_validation/`：约 2.21 GB
- `output_hl/`：约 2.11 GB
- `archive_non_mainline_2026_06_17/`：约 0.88 GB
- `Leave_One_experiments/`：约 0.86 GB
- `.git/`：约 0.46 GB
- `backups/`：约 0.25 GB

这些目录大多包含历史实验结果、DSSAT 输入/输出或备份，不能简单删除。

### 3. `git clean -ndX` 的风险

执行 dry-run 后发现，`git clean -ndX` 会列出：

- `.pytest_cache/`
- `DSSAT_auto_validation/FQZ_IC0_windows480_vs_gym_pdi/`
- `DSSAT_auto_validation/HLA_2004/*.MZX`
- `DSSAT_auto_validation/HLA_2004/*.WTH`
- `DSSAT_auto_validation/HLA_2004/HL.SOL`
- `DSSAT_auto_validation/HLA_2004/MZCER048.CUL`
- 以及大量历史 DSSAT 验证目录

这说明 ignored 文件中混有重要研究输入和历史结果。结论：禁止用 `git clean -fdX` 或类似一键清理方式。

## 建议保留的主线文件

当前阶段最有用、后续最应该优先整理并提交 GitHub 的内容是：

### 1. 代码

优先保留最近与“IC=1、DSSAT 管理模式修正、自由时序 PPO、静态基线重建、五情景图”直接相关的脚本，例如：

- `src/run_original_free_timing_maskableppo_ic1_linked_five_site_half_split_036_01.py`
- `src/run_static_level1_four_baseline_rebuild_037_07.py`
- `src/build_fqa_validation_ppo_fixed_baseline_figures_037_08.py`
- 以及 032、036、037 系列中仍被最新结果引用的训练、评估、绘图脚本

旧的 021--031 系列脚本应保留为历史诊断资料，但不应混入当前“可汇报主线”。

### 2. Prompt 与实验记录

应保留：

- `prompts/032_*`
- `prompts/036_*`
- `prompts/037_*`
- `docs/032_*`
- `docs/036_*`
- `docs/037_*`

后续新任务建议继续按“中文 prompt + 中文实验记录”的格式保存，避免复盘困难。

### 3. 结果与图表

建议 GitHub 只提交小体积、可复核的结果：

- 汇总 CSV
- 指标表
- 代表性 PNG/PDF/SVG 图
- 实验记录 markdown
- 代码与配置

不建议提交：

- 大体积模型 `.zip`
- DSSAT runtime 临时目录
- 重复 checkpoint
- 中间 `PlantGro.OUT` / `Summary.OUT` 批量输出
- 缓存目录

如果某些 DSSAT 输出必须保留，应优先压缩归档或保存在本地，不放进 GitHub 主仓库。

## 建议归档但不删除的历史内容

以下内容建议从“当前主线视图”中归档，而不是删除：

- `021_*` 到 `031_*` 的旧 DQN/PPO 诊断结果
- 旧 reward、DQfD、阶段型 Q-network、监督排序尝试
- 已明确失效的 IC=0 或 DSSAT 管理模式错误运行结果
- `*_attempt*`、`*_failed*`、`runtime_train/`、`runtime_eval/` 等中间目录

建议归档目标可以是：

```text
archive_non_mainline_2026_07_28/
```

但由于移动大量结果目录也会改变工作区结构，必须等用户确认后再执行。

## 需要用户确认的事项

1. `my_data/UFGA8201-*.jinja2` 的删除是否确实要提交？
   - 用户已确认：这是有意删除。
   - 原因：训练数据源已切换到 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/`。
   - 后续提交说明中应明确这一点，避免被误解为误删输入模板。

2. GitHub 是否只备份代码、prompt、docs、CSV、代表性图？
   - 推荐：是。
   - 不推荐：把所有 `benchmark_results/` 大目录完整提交。

3. 是否把 021--031 历史实验移入归档目录？
   - 推荐：先不移动，只在文档中标注“历史诊断线”。
   - 如要移动，应先列出待移动目录清单，再确认。

4. 是否新增 `.gitignore` 规则，排除模型、runtime、缓存、大型 DSSAT 输出？
   - 推荐：新增或修订，但先检查当前 `.gitignore`，避免误排除需要提交的小结果。

## 下一步建议

建议按以下顺序清理：

1. 先确认 `my_data/UFGA8201-*.jinja2` 删除是否提交或恢复。
2. 检查 `.gitignore`，只补充明显安全的缓存/模型/runtime 排除规则。
3. 建立当前主线索引文档，说明 032/036/037 各自是什么、哪些结果可信、哪些结果因 IC 或 DSSAT 管理模式问题仅作历史记录。
4. 只 stage 当前主线代码、prompt、docs、小型 CSV 和代表性图。
5. Git 提交前运行一次 `git status --short`，人工确认没有大体积模型或重要数据误删。

## 本次清理动作记录

- 已做：只读盘点 Git 状态、目录体积、ignored dry-run 清单。
- 已做：发现 `git clean -ndX` 会命中 DSSAT 输入和历史结果，因此禁止一键清理。
- 未做：没有删除任何研究数据。
- 未做：没有移动任何目录。
- 未做：没有提交 Git。
