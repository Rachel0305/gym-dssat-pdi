# 003_05_debug_smoke_test_timeouts

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/003_04_observed_year_smoke_test.md
docs/2026-06-05_observed_year_smoke_test_report.md
```

现在执行新的子任务：排查 `003_04_observed_year_smoke_test` 中出现的 LCA、SYA、YCA 超时问题。

本阶段仍然不要训练 PPO。

---

## 1. 本阶段背景

上一阶段完成了实测年份 smoke test。结果如下：

```text
总 episode 数：56
成功：27
超时：29
```

按站点统计：

```text
FQA: 8 ok, 0 timeout
HLA: 12 ok, 0 timeout
LCA: 4 ok, 12 timeout
SYA: 3 ok, 9 timeout
YCA: 0 ok, 8 timeout
```

已知现象：

1. FQA 全部策略成功；
2. HLA 全部策略成功；
3. LCA 的 `null_zero` 全部成功，但 `fixed_low_input`、`fixed_medium_input`、`fixed_high_input` 全部 timeout；
4. SYA 的 `null_zero` 全部成功，但 `fixed_low_input`、`fixed_medium_input`、`fixed_high_input` 全部 timeout；
5. YCA 的 `null_zero` 和所有固定策略全部 timeout；
6. 成功 episode 的 daily CSV 字段完整；
7. 因此当前不能进入 PPO 训练脚本生成。

本阶段目标是：找出 timeout 的具体原因，并修复到至少所有真实年份的 `null_zero` 和固定策略都能稳定跑完。

---

## 2. 禁止事项

1. 不要训练 PPO。
2. 不要修改 reward 函数。
3. 不要删除、移动、覆盖 `my_data/` 原始文件。
4. 不要直接覆盖原始 `.SOL`、`.jinja2`、`.CUL`、`.WTH` 文件。
5. 不要长时间批量重跑全部实验，必须先做单案例最小复现。
6. 不要用“延长 timeout”掩盖问题。可以临时测试 600s，但必须先判断为什么 300s 超时。
7. 不要静默跳过失败案例。
8. 不要把失败案例从汇总表里删除。

---

## 3. 输入文件

优先读取：

```text
Leave_One_experiments/smoke_tests/evaluation/smoke_test_summary.csv
Leave_One_experiments/smoke_tests/evaluation/smoke_test_minimal_summary.csv
Leave_One_experiments/smoke_tests/daily_outputs/
Leave_One_experiments/smoke_tests/rendered_inputs/
Leave_One_experiments/smoke_tests/logs/
docs/2026-06-05_observed_year_smoke_test_report.md
```

同时需要检查：

```text
data/observed_phenology_dates_standardized.csv
Leave_One_experiments/year_classification/observed_phenology_rainfall_diagnosis.csv
Leave_One_experiments/wth_generated_qc/
weather_clean_qc/
my_data/
```

如果路径不同，请在项目中搜索对应文件名。

---

## 4. 本阶段输出目录

所有排错结果保存到：

```text
Leave_One_experiments/smoke_tests_debug/
```

建议目录结构：

```text
Leave_One_experiments/smoke_tests_debug/
  logs/
  step_traces/
  rendered_inputs_review/
  single_case_tests/
  fixed_outputs/
  evaluation/
  figures/
  reports/
```

任务记录保存到：

```text
docs/2026-06-05_smoke_test_timeout_debug_report.md
docs/2026-06-05_smoke_test_timeout_debug_report.pptx
```

如果日期不方便自动获取，可以使用当前系统日期。

---

## 5. 总体排错思路

请按下面顺序排查，不要跳步：

```text
Step 1. 汇总 timeout 案例，确认每个失败案例的站点、年份、策略、最后输出位置。
Step 2. 选择最小失败案例复现。
Step 3. 对失败案例增加逐步日志，找出卡在哪一天 / 哪个动作 / 哪个环境调用。
Step 4. 检查动作是否正确传入，是否超过 action_space，是否归一化错误。
Step 5. 检查对应渲染后的 DSSAT 输入文件。
Step 6. 检查 DSSAT 输出文件、错误文件和临时目录。
Step 7. 区分是 Python wrapper 卡住、PDI 交互卡住、DSSAT 模型卡住，还是输入文件导致模型等待。
Step 8. 修复后先单案例复测。
Step 9. 单案例通过后，按站点逐步复测。
Step 10. 最后重跑失败的 29 个案例，不要重跑全部 56 个，除非确实必要。
```

---

## 6. 最小失败案例选择

请先复现以下案例：

### 6.1 LCA 动作导致超时案例

```text
station = LCA
year = 2010
policy = fixed_low_input
```

因为 LCA 的 `null_zero` 成功，固定策略失败，说明可能与动作传入、施肥/灌溉事件、action normalization 或交互进程有关。

### 6.2 SYA 动作导致超时案例

```text
station = SYA
year = 2014
policy = fixed_low_input
```

因为 SYA 的 `null_zero` 成功，固定策略失败。

### 6.3 YCA 基础环境超时案例

```text
station = YCA
year = 2014
policy = null_zero
```

因为 YCA 连 `null_zero` 都失败，说明更可能是输入文件、模板、WTH、SOL、管理日期、路径、站点代码或环境初始化问题。

必须先分别排查这三个最小案例，不要直接批量跑。

---

## 7. 增加 step-level trace

请为单案例测试增加详细 step trace。

每一步至少记录：

```text
station
year
policy_name
step_index
date
doy
dap
action_dict_before_normalization
normalized_action
real_action
obs_keys
dap_from_obs
topwt
grnwt
xlai
swfac
nstres
totir
tofer
reward
done
elapsed_time_this_step_seconds
elapsed_time_total_seconds
last_successful_stage
```

保存到：

```text
Leave_One_experiments/smoke_tests_debug/step_traces/
```

命名示例：

```text
LCA_2010_fixed_low_input_step_trace.csv
SYA_2014_fixed_low_input_step_trace.csv
YCA_2014_null_zero_step_trace.csv
```

如果某个 step 超过阈值，例如 20 秒，请立即记录：

```text
slow_step_detected = True
```

并保存当前上下文。

---

## 8. 检查动作和 action_space

对 LCA 和 SYA 的固定策略失败案例，重点检查：

1. `amir` 是否存在于 action_space；
2. `anfer` 是否存在于 action_space；
3. action_space 的 low/high 是多少；
4. 固定策略动作是否超过 high；
5. 动作裁剪是否正确；
6. normalized_action 是否在 `[-1, 1]`；
7. real_action 是否为非负；
8. 施肥动作是否在不允许施肥的日期触发；
9. 灌溉动作是否在不允许灌溉的日期触发；
10. action dict 的 key 是否和环境要求一致；
11. `fixed_low_input` 是否真的只在 DAP 1 和 30 施肥，而不是错误地每天施肥；
12. `fixed_medium_input`、`fixed_high_input` 是否同理。

请输出动作检查表：

```text
Leave_One_experiments/smoke_tests_debug/evaluation/action_space_and_policy_check.csv
```

字段至少包括：

```text
station
year
policy_name
action_name
action_space_low
action_space_high
scheduled_dap
scheduled_real_action
clipped_real_action
normalized_action
is_within_bounds
notes
```

---

## 9. 检查 rendered DSSAT 输入文件

请对成功和失败案例各选一个进行对比。

成功案例：

```text
HLA 2007 fixed_low_input
```

失败案例：

```text
LCA 2010 fixed_low_input
SYA 2014 fixed_low_input
YCA 2014 null_zero
```

检查它们的 rendered input 文件，包括但不限于：

```text
*.MZX 或 fileX.MZX
*.WTH
*.SOL
*.CUL
```

重点检查：

1. WTH 文件是否存在；
2. WTH 站点代码是否和模板中 WSTA 一致；
3. WTH 文件年份是否覆盖模拟年份；
4. WTH 文件是否有缺失日期；
5. WTH 文件是否包含 SRAD、TMAX、TMIN、RAIN；
6. SOL 文件路径是否正确；
7. soil profile ID 是否和模板一致；
8. CUL 文件路径是否正确；
9. cultivar code 是否存在于 `MZCER048.CUL`；
10. planting date 是否正确；
11. harvest date 是否正确；
12. simulation start date 是否早于 planting date；
13. management section 是否存在异常；
14. 初始条件是否合理；
15. `-99` 是否出现在不应出现的关键位置；
16. 对 YCA，重点检查站点代码、模板、SOL、WTH、日期和路径是否一致。

请输出文件检查表：

```text
Leave_One_experiments/smoke_tests_debug/evaluation/rendered_input_check.csv
```

字段至少包括：

```text
case_id
station
year
policy_name
file_type
file_path
exists
check_item
status
details
```

---

## 10. 检查 DSSAT 日志和临时目录

对 timeout 案例，请查找临时运行目录中的 DSSAT 输出和日志文件。

可能包括：

```text
*.OUT
ERROR.OUT
WARNING.OUT
Summary.OUT
PlantGro.OUT
SoilNi.OUT
SoilWat.OUT
Evaluate.OUT
```

如果找不到，请报告临时目录位置和实际存在的文件列表。

请提取最后 50 行或最后可读内容，保存到：

```text
Leave_One_experiments/smoke_tests_debug/logs/
```

注意不要复制过大的完整文件。

请输出日志检查表：

```text
Leave_One_experiments/smoke_tests_debug/evaluation/dssat_log_check.csv
```

字段至少包括：

```text
case_id
station
year
policy_name
log_file
exists
last_lines_path
detected_error_keywords
detected_warning_keywords
notes
```

---

## 11. 可能原因假设

请逐项验证以下假设，不要只猜：

### 假设 A：固定策略动作每天重复触发

表现：

```text
实际 total_n 或 total_irrigation 远大于预期
```

检查：

```text
daily CSV 或 step trace 中 real_action_anfer / real_action_amir 是否仅在指定 DAP 非零
```

### 假设 B：动作归一化或反归一化错误

表现：

```text
normalized_action 超出 [-1, 1]
real_action 变成异常大值或负值
```

检查：

```text
action_space low/high
normalized_action
real_action
传入 env.step 的 action
```

### 假设 C：施肥/灌溉动作 key 不匹配

表现：

```text
env 接收到错误 key 或忽略动作，PDI 交互等待
```

检查：

```text
action_space.keys()
action dict keys
wrapper action_names
```

### 假设 D：YCA 输入文件路径或站点代码不一致

表现：

```text
YCA null_zero 都 timeout
```

检查：

```text
YCA jinja2
YCA SOL
YCA WTH
WSTA
soil profile id
fileX rendered input
```

### 假设 E：某些管理日期超出天气文件覆盖范围

表现：

```text
DSSAT 等待或输出异常
```

检查：

```text
simulation start date
planting date
harvest date
weather start/end date
```

### 假设 F：DSSAT 子进程未正确关闭或等待输入

表现：

```text
Python 层 timeout，但 DSSAT 可能已报错或等待
```

检查：

```text
subprocess 调用
PDI server/client 状态
临时目录输出
```

---

## 12. 修复原则

1. 小修优先，不做大重构；
2. 每次只修一个问题；
3. 修复前备份相关脚本到 `backups/`；
4. 所有修复写入报告；
5. 不要为了通过测试而删除失败站点；
6. 不要简单增大 timeout 代替修复；
7. 修复后必须重新跑对应最小失败案例；
8. 最小失败案例通过后，再重跑该站点所有失败案例；
9. 只在必要时重跑全部 smoke test。

---

## 13. 复测要求

修复后按以下顺序复测：

### 13.1 单案例复测

```text
LCA 2010 fixed_low_input
SYA 2014 fixed_low_input
YCA 2014 null_zero
```

### 13.2 站点级复测

如果单案例通过，再分别复测：

```text
LCA 所有年份 × fixed_low/medium/high
SYA 所有年份 × fixed_low/medium/high
YCA 所有年份 × null_zero/fixed_low/fixed_medium/fixed_high
```

### 13.3 最终复测失败案例

重跑上一阶段失败的 29 个案例，并生成修正版汇总表：

```text
Leave_One_experiments/smoke_tests_debug/evaluation/smoke_test_timeout_rerun_summary.csv
```

字段沿用原始 `smoke_test_summary.csv`，并增加：

```text
original_run_status
rerun_status
fix_applied
debug_notes
```

---

## 14. 图和 daily output

复测成功的案例仍然必须保存：

```text
daily CSV
step trace CSV
response figures
summary row
```

输出到：

```text
Leave_One_experiments/smoke_tests_debug/fixed_outputs/
Leave_One_experiments/smoke_tests_debug/figures/
```

不要覆盖上一阶段原始 smoke test 结果。

---

## 15. 报告输出

请生成 Markdown 报告：

```text
docs/2026-06-05_smoke_test_timeout_debug_report.md
```

报告至少包括：

1. 上一阶段 timeout 概况；
2. 最小失败案例复现结果；
3. step trace 发现；
4. action_space 和固定策略检查；
5. rendered input 检查；
6. DSSAT 日志检查；
7. 每个假设是否成立；
8. 实际原因；
9. 采取的修复；
10. 修复后单案例复测结果；
11. 修复后站点级复测结果；
12. 29 个失败案例重跑结果；
13. 是否可以进入 PPO 训练脚本生成；
14. 如果仍不能进入 PPO，列出剩余阻塞问题。

---

## 16. PPT 输出

请生成或更新 PPT：

```text
docs/2026-06-05_smoke_test_timeout_debug_report.pptx
```

PPT 至少包括：

1. 问题背景；
2. timeout 分布；
3. LCA/SYA/YCA 最小失败案例；
4. action_space 检查；
5. rendered input 检查；
6. 日志检查；
7. 原因定位；
8. 修复方法；
9. 复测结果；
10. 下一步建议。

同时复制一份到：

```text
Leave_One_experiments/smoke_tests_debug/reports/
```

---

## 17. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/003_05_debug_smoke_test_timeouts.md
git add src/
git add experiments/
git add Leave_One_experiments/smoke_tests_debug/
git add docs/2026-06-05_smoke_test_timeout_debug_report.md
git add docs/2026-06-05_smoke_test_timeout_debug_report.pptx
git commit -m "Debug smoke test timeouts before PPO training"
```

不要强行 push。

如果存在大文件，请先报告，不要 commit 大文件。

---

## 18. 完成后请汇报

完成后请汇报：

1. LCA timeout 的原因；
2. SYA timeout 的原因；
3. YCA timeout 的原因；
4. 修复了哪些代码或配置；
5. 是否修改了任何输入文件；
6. 29 个失败案例重跑后成功/失败数量；
7. 是否仍有站点或年份不能跑通；
8. 是否可以进入 PPO 训练脚本生成；
9. 如果不能，下一步阻塞问题是什么。

---

## 19. 最重要原则

1. 当前阶段是排错，不是训练。
2. 不要跳过失败站点。
3. 不要用延长 timeout 掩盖问题。
4. 不要覆盖上一阶段结果。
5. 先定位，再修复，再复测。
6. 所有结果必须有 CSV、日志、报告和 PPT 记录。
