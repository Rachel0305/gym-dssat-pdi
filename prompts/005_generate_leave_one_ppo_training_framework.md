# 003_06_generate_leave_one_ppo_training_framework

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/003_02_generate_wth_and_classify_years.md
prompts/003_03_fix_rain_qc_and_reclassify_years.md
prompts/003_04_observed_year_smoke_test.md
prompts/003_05_debug_smoke_test_timeouts.md
docs/2026-06-05_observed_year_smoke_test_report.md
docs/2026-06-05_smoke_test_timeout_debug_report.md
```

现在执行新的子任务：基于已经跑通的实测年份 smoke test，生成 PPO 训练、验证、绘图框架。

本阶段的目标是生成稳定、可复现、可逐站点调试的 PPO 训练框架。不要一上来批量训练所有站点。

---

## 1. 本阶段背景

上一阶段 `003_05_debug_smoke_test_timeouts` 已经完成 timeout 排查，并确认：

1. 上一阶段 29 个 timeout 案例已经全部重跑成功；
2. LCA、SYA、YCA 的 timeout 根因已定位；
3. smoke test 层面没有剩余阻塞问题；
4. 可以进入 PPO 训练脚本生成；
5. 但是 PPO 训练/评估脚本必须继承 smoke test 中的临时 rendered input 修复逻辑。

特别重要：

PPO 训练和评估脚本必须使用与 smoke test debug 一致的安全渲染逻辑，包括：

1. 不覆盖 `my_data/` 原始模板；
2. 每个站点-年份单独生成临时 rendered input；
3. 临时 rendered input 中必须保证 MI/MF 与 irrigation/fertilizer sections 可用；
4. 清除或替换跨年份残留的静态灌溉/施肥事件；
5. 加入安全的零灌溉/零施肥基线行；
6. PPO 训练前先对对应站点-年份做一次 NullAgent / fixed 策略小 smoke test。

---

## 2. 本阶段不要做的事情

1. 不要批量训练全部 PPO。
2. 不要直接开始五个站点全部训练。
3. 不要修改 reward 函数，除非只是记录当前 reward 版本。
4. 不要修改 `my_data/` 原始数据。
5. 不要直接覆盖原始 `.SOL`、`.jinja2`、`.CUL`、`.WTH` 文件。
6. 不要删除上一阶段 smoke test 和 debug 结果。
7. 不要把 FQA/YCA 只有 2 年真实记录的站点强行称为 dry/normal/wet 三类完整训练。
8. 不要把长期历史代表年和实测年份混在一起训练，除非明确标记为 scenario。
9. 不要只写一个巨大的训练脚本反复改参数。
10. 不要跳过训练前 smoke test。

---

## 3. 本阶段输入

优先使用以下文件：

```text
data/observed_phenology_dates_standardized.csv
Leave_One_experiments/year_classification/observed_phenology_rainfall_diagnosis.csv
Leave_One_experiments/year_classification/observed_phenology_rainfall_rank_by_station.csv
weather_clean_qc/
Leave_One_experiments/wth_generated_qc/
Leave_One_experiments/smoke_tests_debug/evaluation/smoke_test_timeout_rerun_summary.csv
docs/2026-06-05_smoke_test_timeout_debug_report.md
my_data/
MZCER048.CUL
```

如果路径不同，请搜索文件名，不要猜。

---

## 4. 本阶段输出目录

PPO 实验大目录使用：

```text
Leave_One_experiments/ppo_observed_years/
```

建议目录结构：

```text
Leave_One_experiments/ppo_observed_years/
  configs/
  rendered_inputs/
  smoke_checks/
  models/
  logs/
  tensorboard/
  daily_outputs/
  evaluation/
  figures/
  strategy_selection/
  reports/
```

任务记录必须保存到 `docs/`：

```text
docs/2026-06-05_ppo_observed_year_training_framework.md
docs/2026-06-05_ppo_observed_year_training_framework.pptx
```

如果日期不方便自动获取，可以使用当前系统日期。

---

## 5. 站点年份设计原则

本阶段只基于真实实测物候年份，不使用长期历史情景年份。

已通过 QC 的实测年份为：

```text
FQA: 2008, 2010
HLA: 2007, 2011, 2009
LCA: 2010, 2011, 2008, 2009
SYA: 2014, 2015, 2012
YCA: 2014, 2008
```

根据 Phase 2c 诊断：

1. HLA 有 3 年，可做 observed low / mid / high leave-one-year 训练验证；
2. SYA 有 3 年，可做 observed low / mid / high leave-one-year 训练验证；
3. LCA 有 4 年，可做 4-fold observed-year leave-one-out；
4. FQA 只有 2 年，只做 two-year cross validation；
5. YCA 只有 2 年，只做 two-year cross validation；
6. 不要把 FQA/YCA 强行写成三类 dry/normal/wet。

请生成一个统一实验设计表：

```text
Leave_One_experiments/ppo_observed_years/configs/ppo_observed_year_experiment_plan.csv
```

字段至少包括：

```text
station
experiment_group
train_year
train_year_label
validation_years
validation_year_labels
num_observed_years
cross_validation_type
notes
```

其中 `cross_validation_type` 可用：

```text
three_year_leave_one
four_year_leave_one
two_year_cross_validation
```

---

## 6. 推荐训练-验证组合

请根据实测年份降雨排序自动生成组合。

### 6.1 HLA

实测年份：

```text
2007 observed_low_rain_year
2011 observed_mid_rain_year
2009 observed_high_rain_year
```

组合：

```text
train 2007 -> validate 2011, 2009
train 2011 -> validate 2007, 2009
train 2009 -> validate 2007, 2011
```

### 6.2 SYA

实测年份：

```text
2014 observed_low_rain_year
2015 observed_mid_rain_year
2012 observed_high_rain_year
```

组合：

```text
train 2014 -> validate 2015, 2012
train 2015 -> validate 2014, 2012
train 2012 -> validate 2014, 2015
```

### 6.3 LCA

实测年份：

```text
2010 observed_low_rain_year
2011 observed_mid_rain_year
2008 observed_intermediate_rain_year_3
2009 observed_high_rain_year
```

组合：

```text
train 2010 -> validate 2011, 2008, 2009
train 2011 -> validate 2010, 2008, 2009
train 2008 -> validate 2010, 2011, 2009
train 2009 -> validate 2010, 2011, 2008
```

### 6.4 FQA

实测年份：

```text
2008 observed_lower_rain_year
2010 observed_higher_rain_year
```

组合：

```text
train 2008 -> validate 2010
train 2010 -> validate 2008
```

### 6.5 YCA

实测年份：

```text
2014 observed_lower_rain_year
2008 observed_higher_rain_year
```

组合：

```text
train 2014 -> validate 2008
train 2008 -> validate 2014
```

---

## 7. 随机种子设置

当前阶段先只使用一个随机种子：

```text
seed = 0
```

所有代码保留多 seed 扩展接口，但不要默认运行多 seed。

报告中必须说明：

```text
本阶段使用单 seed 进行流程验证和策略初筛，后续可扩展到多 seed 稳定性分析。
```

---

## 8. 训练规模控制

本阶段先生成完整框架，但只运行一个最小 PPO 试训。

建议最小试训：

```text
station = HLA
train_year = 2007
seed = 0
total_timesteps = small_debug_value
```

`small_debug_value` 可以从配置文件控制，例如：

```text
total_timesteps_debug = 1000 或 5000
```

不要一开始运行全部：

```text
HLA 3 + SYA 3 + LCA 4 + FQA 2 + YCA 2 = 14 个 PPO 模型
```

完整批量训练脚本可以生成，但默认不要自动运行全部。

---

## 9. 代码组织要求

请新增或更新以下代码结构。

通用代码放在：

```text
src/
```

建议新增：

```text
src/ppo_experiment_plan.py
src/ppo_safe_rendering.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_plot_results.py
src/ppo_strategy_selection.py
```

站点入口脚本放在：

```text
experiments/ppo_observed_years/
```

建议生成：

```text
experiments/ppo_observed_years/config_ppo_observed_years.yaml
experiments/ppo_observed_years/generate_experiment_plan.py
experiments/ppo_observed_years/train_one_policy.py
experiments/ppo_observed_years/evaluate_one_policy.py
experiments/ppo_observed_years/run_debug_hla_2007.py
experiments/ppo_observed_years/run_all_trainings_DISABLED_BY_DEFAULT.py
experiments/ppo_observed_years/run_all_evaluations_DISABLED_BY_DEFAULT.py
```

注意：

1. 批量脚本文件名必须包含 `DISABLED_BY_DEFAULT`；
2. 批量脚本默认只打印计划，不直接训练；
3. 真正训练需要用户手动修改配置或命令参数；
4. 避免误启动大量 PPO 训练。

---

## 10. 必须继承 smoke test debug 的安全 rendered input 逻辑

这是本阶段最重要要求。

请把 `src/run_smoke_tests.py` 中已经验证有效的安全渲染逻辑抽取到可复用模块：

```text
src/ppo_safe_rendering.py
```

或通用名称：

```text
src/safe_dssat_rendering.py
```

PPO 训练、PPO 评估、NullAgent 检查、固定策略检查都必须调用同一个安全渲染函数。

安全渲染函数必须做到：

1. 不修改 `my_data/` 原始文件；
2. 每次运行生成临时 rendered input；
3. 使用正确站点、正确年份、正确 WTH；
4. 保证 MI/MF 可用；
5. 保证 irrigation section 存在；
6. 保证 fertilizer section 存在；
7. 插入安全的零灌溉行；
8. 插入安全的零施肥行；
9. 清除或替换模拟开始日期之前的历史 irrigation/fertilizer 事件；
10. simulation start date 早于或等于 planting date；
11. weather year 覆盖模拟年份；
12. 渲染结果保存到 `Leave_One_experiments/ppo_observed_years/rendered_inputs/`；
13. 输出 rendered input check 表。

---

## 11. PPO 训练前检查

每个 PPO 训练任务开始前，必须自动执行或要求执行一个小检查：

```text
same station
same train_year
null_zero one-episode check
fixed_low_input one-episode check
```

如果这两个检查任一失败，不允许开始 PPO 训练。

检查结果保存到：

```text
Leave_One_experiments/ppo_observed_years/smoke_checks/
```

汇总表：

```text
Leave_One_experiments/ppo_observed_years/smoke_checks/pretrain_smoke_check_summary.csv
```

字段至少包括：

```text
station
train_year
policy_name
run_status
episode_completed
error_message
daily_csv_path
notes
```

---

## 12. PPO 配置要求

生成配置文件：

```text
experiments/ppo_observed_years/config_ppo_observed_years.yaml
```

至少包含：

```yaml
seed: 0

debug:
  enabled: true
  station: HLA
  train_year: 2007
  total_timesteps: 5000

ppo:
  total_timesteps_full: 100000
  learning_rate: null
  gamma: null
  n_steps: null
  batch_size: null
  ent_coef: null
  clip_range: null

paths:
  output_root: Leave_One_experiments/ppo_observed_years
  weather_dir: weather_clean_qc
  wth_dir: Leave_One_experiments/wth_generated_qc
  my_data_dir: my_data
  cultivar_file: MZCER048.CUL

safety:
  run_pretrain_smoke_check: true
  do_not_overwrite_raw_inputs: true
  batch_training_disabled_by_default: true
```

如果当前项目已有 PPO 参数配置，请读取并写入配置；如果无法确定，就保留为 `null` 并在报告中说明需要用户确认。

不要擅自发明最终 PPO 超参数。

---

## 13. PPO daily output 要求

每次训练完成后，在训练年份上评估一次并保存 daily output。

每次验证年份也必须保存 daily output。

daily output 保存目录：

```text
Leave_One_experiments/ppo_observed_years/daily_outputs/{station}/
```

命名示例：

```text
HLA_train2007_eval2011_seed0_daily.csv
HLA_train2007_eval2009_seed0_daily.csv
```

字段至少包括：

```text
station
policy_name
train_year
train_year_label
eval_year
eval_year_label
seed
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

如果环境没有 `tofer`，使用：

```text
tofer = cumulative_sum(real_action_anfer)
```

如果环境没有 `totir`，使用：

```text
totir = cumulative_sum(real_action_amir)
```

如果环境已有原始 `totir`，同时保存：

```text
totir_raw
totir
```

---

## 14. PPO 评估 summary 要求

每次 `train_year -> eval_year` 输出一行 summary。

保存到：

```text
Leave_One_experiments/ppo_observed_years/evaluation/ppo_evaluation_summary.csv
```

字段至少包括：

```text
station
policy_name
train_year
train_year_label
eval_year
eval_year_label
seed
model_path
run_status
error_message
episode_completed
episode_length
final_grnwt
final_topwt
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

---

## 15. 绘图要求

每个 PPO evaluation 必须生成：

```text
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
crop_growth_timeseries.png
cumulative_water_nitrogen.png
daily_actions.png
```

保存到：

```text
Leave_One_experiments/ppo_observed_years/figures/{station}/{policy_name}/eval_{eval_year}/
```

每个站点后续需要生成策略对比图，但本阶段可先生成脚本，不必批量运行全部训练后才能画。

---

## 16. 策略选择脚本

请生成策略选择脚本：

```text
src/ppo_strategy_selection.py
```

用于在每个站点训练完成后选择最稳定策略。

但本阶段只生成脚本，不要求有完整结果。

选择逻辑：

1. 对同一站点的多个 train_year 策略；
2. 比较其验证年份平均产量、reward、水氮投入和年际波动；
3. 计算 stability_score。

建议公式：

```text
score =
  + normalized_mean_yield
  + normalized_mean_reward
  - normalized_yield_std
  - normalized_total_irrigation
  - normalized_total_n_fertilizer
```

对于 FQA/YCA 只有 2 年的站点，应在输出中标记：

```text
limited_two_year_cross_validation
```

输出文件：

```text
Leave_One_experiments/ppo_observed_years/strategy_selection/best_policy_by_site.csv
```

---

## 17. 最小 PPO 试训

框架生成后，只运行一个最小 PPO 试训：

```text
station = HLA
train_year = 2007
seed = 0
total_timesteps = config.debug.total_timesteps
```

要求：

1. 训练前先跑 HLA 2007 null_zero 和 fixed_low_input pretrain smoke check；
2. 如果 pretrain smoke check 失败，停止；
3. 如果通过，训练小步数 PPO；
4. 保存模型；
5. 在 train_year = 2007 上评估一次；
6. 在 validation_years = 2011, 2009 上各评估一次；
7. 保存 daily CSV、summary、图；
8. 生成 debug 报告。

模型保存路径：

```text
Leave_One_experiments/ppo_observed_years/models/HLA/HLA_train2007_seed0_debug.zip
```

---

## 18. 报告输出

请生成 Markdown 报告：

```text
docs/2026-06-05_ppo_observed_year_training_framework.md
```

报告至少包括：

1. 本阶段目标；
2. 为什么现在可以进入 PPO 框架生成；
3. 为什么还不直接批量训练全部站点；
4. 使用的实测年份；
5. 每个站点训练-验证组合；
6. 安全 rendered input 逻辑如何继承 smoke test debug 修复；
7. 生成了哪些代码；
8. 生成了哪些配置；
9. 最小 PPO 试训是否执行；
10. HLA 2007 debug 训练是否成功；
11. train/eval daily output 是否生成；
12. 训练和评估图是否生成；
13. 当前仍需人工确认的 PPO 超参数；
14. 下一步建议。

---

## 19. PPT 输出

请生成 PPT：

```text
docs/2026-06-05_ppo_observed_year_training_framework.pptx
```

PPT 至少包括：

1. 任务目标；
2. 前置 smoke test 修复结果；
3. PPO 实验设计；
4. 各站点训练-验证组合；
5. 安全 rendered input 机制；
6. pretrain smoke check 机制；
7. PPO daily output 设计；
8. HLA 2007 debug 试训结果；
9. 后续完整训练计划；
10. 风险和注意事项。

并复制一份到：

```text
Leave_One_experiments/ppo_observed_years/reports/
```

---

## 20. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/003_06_generate_leave_one_ppo_training_framework.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/ppo_observed_years/
git add docs/2026-06-05_ppo_observed_year_training_framework.md
git add docs/2026-06-05_ppo_observed_year_training_framework.pptx
git commit -m "Add PPO observed-year leave-one training framework"
```

不要强行 push。

如果存在大模型文件或过大日志，请先报告，不要 commit 大文件。

---

## 21. 完成后请汇报

完成后请汇报：

1. 生成了哪些 PPO 训练/评估/绘图脚本；
2. 生成的训练-验证组合表路径；
3. 安全 rendered input 逻辑是否已复用；
4. HLA 2007 pretrain smoke check 是否通过；
5. HLA 2007 debug PPO 是否成功；
6. 模型保存路径；
7. train/eval daily CSV 是否生成；
8. 图是否生成；
9. 是否可以进入下一步：批量训练 HLA、SYA、LCA；
10. 哪些 PPO 超参数仍需用户确认。

---

## 22. 最重要原则

1. 可以进入 PPO 框架生成，但不要直接批量训练。
2. 先 HLA 2007 小步数 debug。
3. PPO 训练必须继承 smoke test debug 的安全 rendered input 修复逻辑。
4. 每次训练前必须跑 pretrain smoke check。
5. 所有评估必须保存 daily output。
6. 不覆盖原始数据。
7. 不上传大模型文件到 GitHub。
