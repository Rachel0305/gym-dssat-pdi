# 003_09_run_fqa_yca_two_year_action_safe_cross_validation

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/003_07_debug_ppo_action_scale_and_reward_before_batch_training.md
prompts/003_08_run_action_safe_site_level_ppo_training.md
docs/2026-06-05_ppo_action_scale_and_reward_debug_report.md
docs/2026-06-06_action_safe_site_level_ppo_training_report.md
```

如果报告日期或文件名略有不同，请搜索：

```text
action_safe_site_level_ppo_training_report
```

现在执行新的子任务：在启用 action safety 的前提下，对 FQA 和 YCA 做 two-year cross validation。

本阶段仍然不要做多 seed，不要修改 reward，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

上一阶段已经完成 HLA、SYA、LCA 的 action-safe PPO 小批量训练。结果显示：

1. 只训练了 action-safe PPO；
2. HLA、SYA、LCA 的模型均通过 quality gate；
3. 每个 evaluation 的 total_irrigation 均为 200 mm；
4. 每个 evaluation 的 total_n_fertilizer 均为 300 kg/ha；
5. 站点级 best policy 已经输出；
6. FQA/YCA 尚未训练，因为它们只有 2 年实测记录，不能做完整的三年 low/mid/high leave-one 验证。

本阶段目标是：

1. 对 FQA 和 YCA 进行 two-year cross validation；
2. 每个站点训练 2 个 action-safe PPO 模型；
3. 每个模型训练后在训练年份和另一个验证年份上评估；
4. 保存 daily CSV、summary、图、策略选择结果、报告和 PPT；
5. 明确标注 FQA/YCA 是 `limited_two_year_cross_validation`，不要将结果解释为完整 dry/normal/wet 稳定性验证；
6. 检查 action safety 是否仍然导致所有策略刚好打满 200 mm / 300 kg/ha 上限。

---

## 2. 本阶段不要做的事情

1. 不要训练无 action safety 的 PPO。
2. 不要修改 reward。
3. 不要做多 seed。
4. 不要修改 `my_data/` 原始文件。
5. 不要覆盖 HLA/SYA/LCA 的训练结果。
6. 不要把 FQA/YCA 的 2 年交叉验证解释为完整 dry/normal/wet 三类策略稳定性分析。
7. 不要把 action safety 的 200 mm / 300 kg/ha 当作论文最终农学参数。
8. 不要默认 commit 大模型、tensorboard 大日志或过大的 daily outputs。
9. 不要直接进入长时间全量训练。

---

## 3. 本阶段输入

优先读取：

```text
Leave_One_experiments/ppo_observed_years/configs/ppo_observed_year_experiment_plan.csv
Leave_One_experiments/ppo_action_safe_site_training/evaluation/action_safe_site_ppo_evaluation_summary.csv
Leave_One_experiments/ppo_action_safe_site_training/strategy_selection/best_policy_by_site.csv
experiments/ppo_observed_years/config_ppo_action_safe_site_training.yaml
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_plot_results.py
src/ppo_strategy_selection.py
src/ppo_safe_rendering.py
weather_clean_qc/
Leave_One_experiments/wth_generated_qc/
data/observed_phenology_dates_standardized.csv
Leave_One_experiments/year_classification/observed_phenology_rainfall_rank_by_station.csv
```

如果路径不同，请搜索文件名，不要猜。

---

## 4. 本阶段输出目录

本阶段结果保存到新的目录，不覆盖前面结果：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/
```

建议目录结构：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/
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

报告保存到：

```text
docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.md
docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.pptx
```

如果日期不方便自动获取，可以使用当前系统日期。

---

## 5. 本阶段训练站点和组合

本阶段只训练 FQA 和 YCA。

### 5.1 FQA

```text
FQA_train2008_seed0_action_safe
  train_year = 2008
  train_year_label = observed_lower_rain_year
  validate = 2010
  eval_years = 2008, 2010

FQA_train2010_seed0_action_safe
  train_year = 2010
  train_year_label = observed_higher_rain_year
  validate = 2008
  eval_years = 2010, 2008
```

### 5.2 YCA

```text
YCA_train2014_seed0_action_safe
  train_year = 2014
  train_year_label = observed_lower_rain_year
  validate = 2008
  eval_years = 2014, 2008

YCA_train2008_seed0_action_safe
  train_year = 2008
  train_year_label = observed_higher_rain_year
  validate = 2014
  eval_years = 2008, 2014
```

合计：

```text
FQA 2 个模型
YCA 2 个模型
共 4 个 PPO 模型
```

---

## 6. 运行顺序

请严格按以下顺序：

### 6.1 生成本阶段配置

生成：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/configs/two_year_cv_training_plan.csv
experiments/ppo_observed_years/config_ppo_action_safe_two_year_cv.yaml
```

### 6.2 先跑 FQA 两个模型

顺序：

```text
FQA_train2008_seed0_action_safe
FQA_train2010_seed0_action_safe
```

如果 FQA 任一模型失败或 quality gate 不通过，停止，不要继续 YCA。

### 6.3 FQA 全部通过后，再跑 YCA 两个模型

顺序：

```text
YCA_train2014_seed0_action_safe
YCA_train2008_seed0_action_safe
```

如果 YCA 任一模型失败或 quality gate 不通过，停止并报告。

---

## 7. 训练设置

沿用上一阶段 action-safe site-level PPO 的配置。

建议：

```yaml
seed: 0

training:
  total_timesteps: 5000
  save_model: true

action_safety:
  enabled: true
  daily_irrigation_max: 40
  daily_n_max: 80
  season_irrigation_soft_limit: 200
  season_n_soft_limit: 300
  min_days_between_irrigation: 7
  min_days_between_fertilization: 10
  fertilization_allowed_dap_min: 1
  fertilization_allowed_dap_max: 90
  irrigation_allowed_dap_min: 1
  irrigation_allowed_dap_max: 120

quality_gates:
  max_total_irrigation: 300
  max_total_n_fertilizer: 400
  require_daily_csv: true
  require_figures: true
  stop_on_failed_episode: true
  stop_on_quality_gate_failure: true
```

如果上一阶段实际使用的是 `total_timesteps = 5000`，本阶段也使用 5000，以保证可比性。

---

## 8. 每个模型训练前必须做 pretrain smoke check

每个模型训练前，必须对同一个 station 和 train_year 做：

```text
null_zero
fixed_low_input
```

保存到：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/smoke_checks/
```

汇总表：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/smoke_checks/pretrain_smoke_check_summary.csv
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

如果任一 pretrain smoke check 失败，停止该模型训练，并写入报告。

---

## 9. 每个模型训练后评估要求

每个模型训练完成后，必须评估：

1. train_year 本身；
2. 另一个 validation year。

例如：

```text
FQA_train2008_seed0_action_safe
  eval 2008
  eval 2010
```

每个 eval 都必须保存 daily CSV、summary 和图。

---

## 10. daily output 字段要求

保存目录：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/daily_outputs/{station}/
```

命名示例：

```text
FQA_train2008_eval2010_seed0_action_safe_daily.csv
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
raw_real_action_amir
raw_real_action_anfer
safe_real_action_amir
safe_real_action_anfer
real_action_amir
real_action_anfer
normalized_action_amir
normalized_action_anfer
action_clipped_amir
action_clipped_anfer
season_irrigation_so_far
season_n_so_far
safety_rule_triggered
done
info
```

如果某些字段暂时无法获得，请在报告中明确说明，不要静默缺失。

---

## 11. evaluation summary 要求

保存总表：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/evaluation/two_year_cv_ppo_evaluation_summary.csv
```

每个 `train_year -> eval_year` 一行。

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
quality_gate_pass
quality_gate_reason
notes
```

---

## 12. quality gate

每个 evaluation 都必须检查：

```text
episode_completed == True
run_status == ok
total_irrigation <= 300
total_n_fertilizer <= 400
daily_csv_path exists
figure_dir exists
```

如果任一不满足：

```text
quality_gate_pass = False
```

并记录：

```text
quality_gate_reason
```

如果某个模型任一 eval 年份不通过 quality gate，停止后续训练，并写入报告。

---

## 13. 额外诊断：是否打满 action safety 上限

上一阶段 HLA/SYA/LCA 所有模型的 mean_irrigation 和 mean_n 均为：

```text
mean_irrigation = 200
mean_n = 300
```

这说明 action safety 约束可能正在主导策略行为。

本阶段必须额外输出：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/evaluation/action_safety_saturation_check.csv
```

字段至少包括：

```text
station
policy_name
train_year
eval_year
total_irrigation
total_n_fertilizer
irrigation_at_season_limit
n_at_season_limit
num_irrigation_clipped_days
num_n_clipped_days
num_safety_trigger_days
dominant_safety_rule
notes
```

如果 FQA/YCA 也全部打满 200/300，请在报告中明确说明：

```text
当前 action-safe PPO 的策略比较更多反映了固定 season cap 下的时序分配差异，而不是自由水氮优化；下一阶段需要考虑 reward 成本项或 season cap 敏感性分析。
```

---

## 14. 图表输出

每个 eval 至少生成：

```text
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
crop_growth_timeseries.png
cumulative_water_nitrogen.png
daily_actions_raw_vs_safe.png
action_safety_triggers.png
```

保存到：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/figures/{station}/{policy_name}/eval_{eval_year}/
```

每个站点完成后生成站点级对比图：

```text
{station}_final_grnwt_by_policy_eval_year.png
{station}_total_irrigation_by_policy_eval_year.png
{station}_total_n_by_policy_eval_year.png
{station}_mean_reward_by_policy_eval_year.png
{station}_stability_score_by_policy.png
{station}_action_safety_saturation.png
```

---

## 15. 策略选择

每个站点全部模型完成后，运行：

```text
src/ppo_strategy_selection.py
```

输出：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/strategy_selection/best_policy_by_site.csv
```

评分逻辑沿用上一阶段，但必须标记：

```text
limited_two_year_cross_validation
```

同时输出：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/strategy_selection/policy_ranking_by_site.csv
```

注意：

FQA/YCA 只有 2 年，策略选择只是初步 two-year 结果，不应与 HLA/SYA/LCA 的 3-4 年稳定性分析等价。

---

## 16. 汇总所有五站点结果

如果 FQA/YCA 全部通过，请生成五站点总表：

```text
Leave_One_experiments/ppo_action_safe_summary/all_site_best_policy_summary.csv
```

内容整合：

```text
HLA/SYA/LCA 来自 ppo_action_safe_site_training
FQA/YCA 来自 ppo_action_safe_two_year_cv
```

字段至少包括：

```text
station
best_policy
train_year
validation_years
cross_validation_type
mean_yield
mean_reward
mean_irrigation
mean_n
stability_score
quality_gate_pass
limited_data_flag
notes
```

其中：

```text
limited_data_flag = True
```

用于 FQA/YCA。

---

## 17. 失败处理

如果训练或评估失败：

1. 保存错误日志；
2. 保存已经完成的结果；
3. 不删除模型；
4. 不覆盖已有 summary；
5. 停止后续模型；
6. 在报告中写明失败站点、年份、模型、错误原因；
7. 给出下一步排错建议。

---

## 18. 报告输出

请生成 Markdown 报告：

```text
docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.md
```

报告至少包括：

1. 本阶段目标；
2. 为什么 FQA/YCA 只能做 two-year cross validation；
3. 训练组合；
4. action safety 参数；
5. 每个模型的 pretrain smoke check 结果；
6. 每个模型训练状态；
7. 每个模型 train/eval 结果；
8. quality gate 检查结果；
9. action safety 是否打满上限；
10. FQA/YCA 策略选择结果；
11. 五站点 best policy 总表；
12. 是否可以进入多 seed 稳定性分析；
13. 是否需要做 season cap 敏感性分析；
14. 是否需要修改 reward 加入成本项。

---

## 19. PPT 输出

请生成 PPT：

```text
docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.pptx
```

PPT 至少包括：

1. 任务目标；
2. FQA/YCA 为什么单独处理；
3. two-year cross validation 设计；
4. FQA 结果；
5. YCA 结果；
6. action safety 是否打满；
7. 五站点 best policy 总表；
8. 当前方法局限；
9. 下一步计划。

并复制一份到：

```text
Leave_One_experiments/ppo_action_safe_two_year_cv/reports/
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
git add prompts/003_09_run_fqa_yca_two_year_action_safe_cross_validation.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/ppo_action_safe_two_year_cv/configs/
git add Leave_One_experiments/ppo_action_safe_two_year_cv/evaluation/
git add Leave_One_experiments/ppo_action_safe_two_year_cv/strategy_selection/
git add Leave_One_experiments/ppo_action_safe_two_year_cv/reports/
git add Leave_One_experiments/ppo_action_safe_summary/
git add docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.md
git add docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.pptx
git commit -m "Run action-safe two-year PPO cross validation for FQA and YCA"
```

注意：

1. 不要默认 commit 大模型 `.zip`；
2. 不要默认 commit tensorboard 大日志；
3. 不要默认 commit 过大的 daily_outputs；
4. 如果需要保存小型 debug 模型，请先报告文件大小；
5. 不要强行 push。

---

## 21. 完成后请汇报

完成后请汇报：

1. FQA 两个模型是否全部完成；
2. YCA 两个模型是否全部完成；
3. 每个模型的 total_irrigation 和 total_n_fertilizer 是否通过 quality gate；
4. FQA 最稳定策略是哪一个；
5. YCA 最稳定策略是哪一个；
6. 五站点 best policy 总表路径；
7. FQA/YCA 是否也打满 200 mm / 300 kg/ha 上限；
8. 是否可以进入多 seed 稳定性分析；
9. 是否需要先做 season cap 敏感性分析或 reward 成本项修正。

---

## 22. 最重要原则

1. 只训练 action-safe PPO。
2. 只训练 FQA/YCA。
3. FQA/YCA 必须标记为 limited two-year cross validation。
4. 每个模型训练前必须 smoke check。
5. 每个模型训练后必须 train/eval 双年份评估。
6. 每个 evaluation 必须保存 daily output。
7. 任一 quality gate 不通过就停止。
8. 不做多 seed。
9. 不覆盖旧结果。
10. 不 commit 大模型和大日志。
