# 003_04_observed_year_smoke_test

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/003_02_generate_wth_and_classify_years.md
prompts/003_03_fix_rain_qc_and_reclassify_years.md
```

并阅读上一阶段生成的诊断报告：

```text
docs/phase2c_observed_phenology_rainfall_diagnosis.md
```

如果项目中该报告路径不同，请搜索文件名：

```text
phase2c_observed_phenology_rainfall_diagnosis.md
```

现在执行新的子任务：对通过 QC 的实测物候年份进行 `NullAgent / 固定水氮策略 smoke test`。

本阶段仍然不要训练 PPO。

---

## 1. 本阶段目标

上一阶段已经确认：

1. 真实试验年份的 `planting_date -> harvest_date` 已整理完成；
2. 日期 QC 通过；
3. RAIN 字段识别正常；
4. 实测物候年份的播种到收获期降雨量已计算；
5. 部分站点只有 2 年真实试验记录，不能强行称为完整 dry / normal / wet；
6. 推荐下一步是先对通过 QC 的实测年份做 `NullAgent` 和固定策略 smoke test。

本阶段目标是：

1. 确认每个站点、每个实测年份都能在 gym-DSSAT 中稳定跑完；
2. 确认 `.WTH`、`.SOL`、`.jinja2`、`MZCER048.CUL`、初始条件能被正确读取；
3. 确认 daily history 能完整保存；
4. 确认 `dap`、`topwt`、`grnwt`、`xlai`、`totir`、`tofer`、`swfac`、`nstres`、`reward`、动作变量可以稳定输出；
5. 用固定策略初步检查水氮动作是否能引起合理的模拟响应；
6. 在进入 PPO 训练前发现并解决环境卡死、路径错误、WTH 格式错误、变量缺失、动作记录错误等问题。

---

## 2. 本阶段不要做的事情

1. 不要训练 PPO。
2. 不要修改 reward 函数。
3. 不要修改 `sb3_wrapper.py`，除非只是为了读取已有变量且必须先备份并说明。
4. 不要删除、移动、覆盖 `my_data/` 中的原始数据。
5. 不要直接覆盖原始 `.SOL`、`.jinja2`、`.CUL`、`.WTH` 文件。
6. 不要进行长时间训练。
7. 不要把 smoke test 结果当作最终优化结果。
8. 不要强行把只有 2 年真实记录的站点称为完整 dry / normal / wet。
9. 不要只打印终端结果，必须保存 CSV、图、Markdown 报告和 PPT 记录。

---

## 3. 输入文件

优先使用以下文件：

```text
data/observed_phenology_dates_standardized.csv
Leave_One_experiments/year_classification/observed_phenology_rainfall_diagnosis.csv
Leave_One_experiments/year_classification/observed_phenology_rainfall_rank_by_station.csv
weather_clean_qc/
Leave_One_experiments/wth_generated_qc/
my_data/
MZCER048.CUL
```

如果 `wth_generated_qc/` 不存在，但上一阶段已经生成了可用 WTH 文件，请在报告中说明实际使用的 WTH 目录。

如果需要临时渲染 `.jinja2` 模板，请只生成到本阶段输出目录，不要覆盖 `my_data/` 原始模板。

---

## 4. 本阶段测试年份

请优先测试 Phase 2c 推荐的实测年份：

```text
FQA: 2008, 2010
HLA: 2007, 2011, 2009
LCA: 2010, 2011, 2008, 2009
SYA: 2014, 2015, 2012
YCA: 2014, 2008
```

这些年份来自 `observed_phenology_dates_standardized.csv` 和 `observed_phenology_rainfall_rank_by_station.csv`。

注意：

1. HLA、SYA 有 3 年，可以作为 observed low / mid / high rainfall 梯度；
2. LCA 有 4 年，可以作为 observed low / mid / intermediate / high rainfall 梯度；
3. FQA、YCA 只有 2 年，只能称为 observed lower / higher rainfall years；
4. 本阶段不使用长期历史 Phase 2b 年份作为训练年份；
5. 本阶段只验证真实试验年份环境是否能跑通。

---

## 5. 输出总目录

本阶段所有结果保存到：

```text
Leave_One_experiments/smoke_tests/
```

建议目录结构：

```text
Leave_One_experiments/smoke_tests/
  configs/
  rendered_inputs/
  logs/
  daily_outputs/
  evaluation/
  figures/
  reports/
```

同时，任务记录必须保存到 `docs/`：

```text
docs/2026-06-05_observed_year_smoke_test_report.md
docs/2026-06-05_observed_year_smoke_test_report.pptx
```

如果日期不方便自动获取，可以使用当前系统日期生成文件名。

---

## 6. 代码组织要求

建议新增或更新以下代码，但不要破坏已有代码：

```text
src/smoke_test_agents.py
src/run_smoke_tests.py
src/plot_smoke_test_results.py
experiments/smoke_tests/run_observed_year_smoke_tests.py
experiments/smoke_tests/config_observed_year_smoke_tests.yaml
```

要求：

1. 通用逻辑放在 `src/`；
2. 入口脚本放在 `experiments/smoke_tests/`；
3. 不要把所有逻辑写成一个难维护的大脚本；
4. 每个站点、年份、策略的结果必须可追踪；
5. 关键路径、策略参数、动作日期、动作量写入 YAML 配置。

---

## 7. Smoke test 策略设计

本阶段策略不是最终推荐管理方案，只是为了检查环境能否稳定运行、动作是否能被执行、输出是否完整。

请实现以下策略。

### 7.1 `null_zero`

每日动作：

```text
amir = 0
anfer = 0
```

用途：

```text
检查不灌溉、不施肥情况下环境是否能跑完。
```

### 7.2 `fixed_low_input`

建议总投入：

```text
total_irrigation = 0 或较低固定值
total_n = 50 kg/ha
```

建议施肥安排：

```text
DAP 1: 30 kg/ha N
DAP 30: 20 kg/ha N
```

如果当前环境从 DAP 0 或 DAP 1 开始不同，请在代码中自动适配第一个可用 DAP。

### 7.3 `fixed_medium_input`

建议总投入：

```text
total_irrigation = 60 mm
total_n = 150 kg/ha
```

建议安排：

```text
DAP 1: 50 kg/ha N
DAP 30: 50 kg/ha N
DAP 60: 50 kg/ha N

DAP 30: 30 mm irrigation
DAP 60: 30 mm irrigation
```

### 7.4 `fixed_high_input`

建议总投入：

```text
total_irrigation = 120 mm
total_n = 250 kg/ha
```

建议安排：

```text
DAP 1: 80 kg/ha N
DAP 30: 80 kg/ha N
DAP 60: 90 kg/ha N

DAP 30: 40 mm irrigation
DAP 60: 40 mm irrigation
DAP 90: 40 mm irrigation
```

### 7.5 重要注意事项

1. 不要每天都施固定氮肥，否则会造成不合理的巨大施氮量；
2. 不要每天都灌固定水量；
3. 固定策略必须是基于指定 DAP 的离散动作；
4. 如果环境 action space 的最大值小于上述动作量，请自动裁剪到 action space 上限，并在日志中记录；
5. 如果当前环境只有 `anfer` 或只有 `amir`，请自动跳过不存在的动作，并在报告中说明；
6. 所有动作都要记录 `real_action_amir`、`real_action_anfer`、`normalized_action_amir`、`normalized_action_anfer`；
7. 如果无法直接获得 normalized action，请至少保存策略输出前的动作值和实际传给环境的动作值，并在报告中说明。

---

## 8. 每次运行必须保存 daily output

每个 `station-year-policy` 都必须保存一个 daily CSV。

保存目录：

```text
Leave_One_experiments/smoke_tests/daily_outputs/{station}/
```

命名示例：

```text
HLA_2007_null_zero_daily.csv
HLA_2007_fixed_low_input_daily.csv
HLA_2007_fixed_medium_input_daily.csv
HLA_2007_fixed_high_input_daily.csv
```

每个 daily CSV 至少包含：

```text
station
year
observed_rain_label
policy_name
date
doy
dap
topwt
grnwt
xlai
totir
tofer
swfac
nstres
reward
real_action_amir
real_action_anfer
normalized_action_amir
normalized_action_anfer
done
info
```

如果环境没有 `tofer`，请用：

```text
tofer = cumulative_sum(real_action_anfer)
```

如果环境没有 `totir`，请用：

```text
totir = cumulative_sum(real_action_amir)
```

如果环境已有原始 `totir`，请同时保存：

```text
totir_raw
totir
```

---

## 9. 每次运行必须保存 summary

每个 `station-year-policy` 输出一行 summary。

保存总表：

```text
Leave_One_experiments/smoke_tests/evaluation/smoke_test_summary.csv
```

字段至少包括：

```text
station
year
observed_rain_label
harvest_window_rain_mm
policy_name
run_status
error_message
episode_completed
episode_length
final_dap
final_topwt
final_grnwt
final_xlai
total_irrigation
total_n_fertilizer
mean_swfac
mean_nstres
mean_reward
sum_reward
daily_csv_path
figure_dir
notes
```

其中：

```text
run_status
```

可取：

```text
ok
failed
timeout
variable_missing
input_missing
```

如果某个环境卡死或超时，请记录为 `timeout`，不要让整个批量任务永久卡住。

---

## 10. 图表输出

每个 `station-year-policy` 至少生成以下图：

保存到：

```text
Leave_One_experiments/smoke_tests/figures/{station}/{year}/{policy_name}/
```

图包括：

```text
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
crop_growth_timeseries.png
cumulative_water_nitrogen.png
daily_actions.png
```

每个站点还要生成对比图：

```text
Leave_One_experiments/smoke_tests/figures/{station}/station_policy_comparison/
```

至少包括：

```text
{station}_final_grnwt_by_year_policy.png
{station}_total_irrigation_by_year_policy.png
{station}_total_n_by_year_policy.png
{station}_mean_swfac_by_year_policy.png
{station}_mean_nstres_by_year_policy.png
```

---

## 11. 环境运行前检查

正式批量运行前，请先执行一个最小 smoke test：

```text
station = HLA
year = 2007
policy = null_zero
```

确认：

1. 环境能创建；
2. `reset()` 能返回 observation；
3. `step()` 能正常运行；
4. episode 能结束；
5. daily CSV 能保存；
6. 图能生成。

如果最小测试失败，请停止批量运行，输出错误报告，不要继续跑所有站点。

---

## 12. 批量运行要求

最小测试通过后，运行：

```text
14 个实测站点-年份 × 4 个固定策略
```

也就是：

```text
FQA 2 年 × 4 策略
HLA 3 年 × 4 策略
LCA 4 年 × 4 策略
SYA 3 年 × 4 策略
YCA 2 年 × 4 策略
```

理论共：

```text
56 次 episode
```

每次 episode 必须独立保存结果，不能互相覆盖。

---

## 13. 报告输出

请生成 Markdown 报告：

```text
docs/2026-06-05_observed_year_smoke_test_report.md
```

报告至少包括：

1. 本阶段目的；
2. 使用的站点和年份；
3. 使用的策略；
4. 每个站点-年份-policy 是否跑通；
5. 失败的案例和错误信息；
6. 变量缺失情况；
7. daily output 保存情况；
8. 图表保存情况；
9. 各固定策略的 final_grnwt、total_irrigation、total_n_fertilizer 对比；
10. 是否出现明显不合理结果；
11. 是否可以进入下一步 PPO 训练脚本生成；
12. 哪些站点或年份需要单独排错。

---

## 14. PPT 输出

请生成或更新 PPT：

```text
docs/2026-06-05_observed_year_smoke_test_report.pptx
```

PPT 至少包括：

1. 任务目标；
2. 输入数据和站点年份；
3. 固定策略设计；
4. 最小 smoke test 结果；
5. 56 次 episode 批量运行状态；
6. 每个站点固定策略产量对比；
7. 每个站点水氮投入对比；
8. 示例 daily response 图；
9. 失败或异常案例；
10. 下一步建议。

同时可以把报告和 PPT 复制一份到：

```text
Leave_One_experiments/smoke_tests/reports/
```

---

## 15. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/003_04_observed_year_smoke_test.md
git add src/
git add experiments/
git add Leave_One_experiments/smoke_tests/
git add docs/2026-06-05_observed_year_smoke_test_report.md
git add docs/2026-06-05_observed_year_smoke_test_report.pptx
git commit -m "Add observed year smoke tests for fixed water nitrogen strategies"
```

不要强行 push。

如果存在大文件，请先报告，不要 commit 大文件。

---

## 16. 完成后请汇报

完成后请汇报：

1. 最小 smoke test 是否通过；
2. 56 次 episode 中多少次成功、多少次失败；
3. 哪些站点、年份、策略失败；
4. 失败原因是什么；
5. daily CSV 是否全部生成；
6. 图是否全部生成；
7. 哪些变量缺失或异常；
8. 固定策略是否能产生不同的水氮投入和作物响应；
9. 是否可以进入下一步 PPO 训练脚本生成。

---

## 17. 最重要原则

1. 这个阶段是环境体检，不是优化。
2. 先跑通，再训练。
3. 所有失败都要记录，不要静默跳过。
4. 所有 episode 都必须保存 daily output。
5. 不要修改原始数据。
6. 不要把固定策略结果解释成最终优化策略。
