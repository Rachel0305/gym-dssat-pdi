# 006_02_season_cap_sensitivity_analysis

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/005_01_debug_ppo_action_scale_and_reward_before_batch_training.md
prompts/006_run_action_safe_site_level_ppo_training.md
prompts/006_01_run_fqa_yca_two_year_action_safe_cross_validation.md
docs/2026-06-06_action_safe_site_level_ppo_training_report.md
docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.md
```

如果文件名或日期略有差异，请搜索以下关键词：

```text
action_safe_site_level_ppo_training_report
fqa_yca_two_year_action_safe_cv_report
```

现在执行新的子任务：进行 season cap 敏感性分析，判断当前 action-safe PPO 是否被固定 season cap 主导。

本阶段不要做多 seed，不要修改 reward，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

前面已经完成：

1. HLA、SYA、LCA 的 action-safe observed-year leave-one PPO；
2. FQA、YCA 的 action-safe two-year cross validation；
3. 五站点 best policy 总表；
4. 所有 evaluation 均通过 quality gate；
5. 但所有站点、所有模型几乎都打满了当前 action safety 上限：

```text
season_irrigation_soft_limit = 200 mm
season_n_soft_limit = 300 kg/ha
```

这说明当前 action-safe PPO 的策略比较可能更多反映：

```text
在固定 200 mm / 300 kg/ha season cap 下，PPO 如何分配水氮时序
```

而不是：

```text
PPO 自由学习到了最优灌溉总量和最优施氮总量
```

因此，本阶段需要系统检查不同 season cap 组合下，PPO 是否仍然总是打满上限、产量是否持续升高、reward 是否合理变化、边际收益是否递减。

---

## 2. 本阶段目标

本阶段目标是回答以下问题：

1. 当前 200 mm / 300 kg/ha cap 是否主导了 PPO 策略？
2. 如果 cap 降低，产量和 reward 会下降多少？
3. 如果 cap 提高，PPO 是否继续打满更高上限？
4. 是否存在产量边际收益递减区间？
5. 是否不同站点对 cap 的敏感性不同？
6. 当前 action safety cap 是否可以作为后续正式实验的合理约束？
7. 是否必须进入下一阶段 reward 成本项修正？

---

## 3. 本阶段不要做的事情

1. 不要训练无 action safety 的 PPO。
2. 不要修改 reward。
3. 不要做多 seed。
4. 不要修改 `my_data/` 原始文件。
5. 不要覆盖前面 `ppo_action_safe_site_training/` 和 `ppo_action_safe_two_year_cv/` 的结果。
6. 不要把 cap 敏感性结果直接解释为最终最优水氮管理。
7. 不要一次性训练过多模型导致无法排错。
8. 不要默认 commit 大模型、tensorboard 大日志或过大的 daily outputs。
9. 不要把 FQA/YCA 的 two-year 结果和 HLA/SYA/LCA 的 3-4 年 leave-one 结果等同解释。

---

## 4. 输入文件

优先读取：

```text
Leave_One_experiments/ppo_action_safe_summary/all_site_best_policy_summary.csv
Leave_One_experiments/ppo_action_safe_site_training/strategy_selection/best_policy_by_site.csv
Leave_One_experiments/ppo_action_safe_two_year_cv/strategy_selection/best_policy_by_site.csv
Leave_One_experiments/ppo_action_safe_site_training/evaluation/action_safe_site_ppo_evaluation_summary.csv
Leave_One_experiments/ppo_action_safe_two_year_cv/evaluation/two_year_cv_ppo_evaluation_summary.csv
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

## 5. 输出目录

本阶段结果保存到新目录，不覆盖前面结果：

```text
Leave_One_experiments/season_cap_sensitivity/
```

建议目录结构：

```text
Leave_One_experiments/season_cap_sensitivity/
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
docs/2026-06-06_season_cap_sensitivity_analysis_report.md
docs/2026-06-06_season_cap_sensitivity_analysis_report.pptx
```

如果日期不方便自动获取，可以使用当前系统日期。

---

## 6. cap 组合设计

本阶段先做有限 cap 组合，不要无限扩大。

建议 cap 组合：

```text
cap_low:
  season_irrigation_soft_limit = 100 mm
  season_n_soft_limit = 150 kg/ha

cap_mid_low:
  season_irrigation_soft_limit = 150 mm
  season_n_soft_limit = 225 kg/ha

cap_current:
  season_irrigation_soft_limit = 200 mm
  season_n_soft_limit = 300 kg/ha

cap_mid_high:
  season_irrigation_soft_limit = 250 mm
  season_n_soft_limit = 375 kg/ha

cap_high:
  season_irrigation_soft_limit = 300 mm
  season_n_soft_limit = 450 kg/ha
```

其他 action safety 参数先保持不变：

```text
daily_irrigation_max = 40 mm
daily_n_max = 80 kg/ha
min_days_between_irrigation = 7
min_days_between_fertilization = 10
fertilization_allowed_dap_range = 1-90
irrigation_allowed_dap_range = 1-120
```

---

## 7. 站点和模型选择

为了控制计算量，本阶段不要对所有 train_year 全部重训。

请优先对五站点 best policy 对应 train_year 做 cap 敏感性分析。

根据前面结果，五站点 best policy 为：

```text
FQA: FQA_train2010_seed0_action_safe
HLA: HLA_train2011_seed0_action_safe
LCA: LCA_train2010_seed0_action_safe
SYA: SYA_train2012_seed0_action_safe
YCA: YCA_train2008_seed0_action_safe
```

本阶段每个站点只训练该 best train_year 在不同 cap 下的 PPO。

对应训练年份和验证年份：

```text
FQA:
  train_year = 2010
  eval_years = 2010, 2008
  cv_type = limited_two_year_cross_validation

HLA:
  train_year = 2011
  eval_years = 2011, 2007, 2009
  cv_type = observed_year_leave_one

LCA:
  train_year = 2010
  eval_years = 2010, 2011, 2008, 2009
  cv_type = observed_year_leave_one

SYA:
  train_year = 2012
  eval_years = 2012, 2014, 2015
  cv_type = observed_year_leave_one

YCA:
  train_year = 2008
  eval_years = 2008, 2014
  cv_type = limited_two_year_cross_validation
```

总训练模型数：

```text
5 sites × 5 cap levels = 25 models
```

如果计算压力太大，请先做三站点 pilot：

```text
HLA, LCA, SYA × 5 cap levels = 15 models
```

但默认建议先按站点顺序逐个跑，而不是一次性跑完 25 个。

---

## 8. 运行顺序

严格按以下顺序：

### 8.1 生成配置和计划表

生成：

```text
Leave_One_experiments/season_cap_sensitivity/configs/season_cap_sensitivity_plan.csv
experiments/ppo_observed_years/config_season_cap_sensitivity.yaml
```

计划表字段至少包括：

```text
station
train_year
train_year_label
eval_years
cv_type
cap_name
season_irrigation_cap
season_n_cap
seed
total_timesteps
run_order
notes
```

### 8.2 先跑 HLA pilot

先只跑：

```text
HLA train_year = 2011
cap_low
cap_mid_low
cap_current
cap_mid_high
cap_high
```

如果 HLA 任一 cap 失败，停止，不要继续其他站点。

### 8.3 HLA 成功后，再跑 SYA 和 LCA

顺序：

```text
SYA train_year = 2012 × 5 caps
LCA train_year = 2010 × 5 caps
```

如果任一站点失败，停止并报告。

### 8.4 三个 3+ 年站点成功后，再跑 FQA/YCA

顺序：

```text
FQA train_year = 2010 × 5 caps
YCA train_year = 2008 × 5 caps
```

FQA/YCA 结果必须标记为 limited two-year cross validation。

---

## 9. 训练设置

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
  min_days_between_irrigation: 7
  min_days_between_fertilization: 10
  fertilization_allowed_dap_min: 1
  fertilization_allowed_dap_max: 90
  irrigation_allowed_dap_min: 1
  irrigation_allowed_dap_max: 120

quality_gates:
  run_status_required: ok
  episode_completed_required: true
  require_daily_csv: true
  require_figures: true
  stop_on_failed_episode: true
  stop_on_quality_gate_failure: true
```

注意：

本阶段不同 cap 的 quality gate 不再固定为：

```text
irrigation <= 300
nitrogen <= 400
```

而应检查是否超过对应 cap 加上很小容差：

```text
total_irrigation <= season_irrigation_cap + 1e-6
total_n_fertilizer <= season_n_cap + 1e-6
```

---

## 10. 每个模型训练前 pretrain smoke check

每个模型训练前，对同一站点、同一 train_year、同一 cap 设置，运行：

```text
null_zero
fixed_low_input
```

保存到：

```text
Leave_One_experiments/season_cap_sensitivity/smoke_checks/
```

汇总表：

```text
Leave_One_experiments/season_cap_sensitivity/smoke_checks/pretrain_smoke_check_summary.csv
```

如果任一 pretrain smoke check 失败，停止该模型训练。

---

## 11. 每个模型训练后评估

每个模型训练完成后，必须在 train_year 和所有 validation years 上评估。

每个 eval 都保存：

```text
daily CSV
summary row
figures
```

daily output 保存到：

```text
Leave_One_experiments/season_cap_sensitivity/daily_outputs/{station}/
```

命名示例：

```text
HLA_train2011_cap_low_eval2007_seed0_daily.csv
```

---

## 12. daily output 字段

字段至少包括：

```text
station
policy_name
cap_name
season_irrigation_cap
season_n_cap
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

---

## 13. evaluation summary

保存总表：

```text
Leave_One_experiments/season_cap_sensitivity/evaluation/season_cap_sensitivity_evaluation_summary.csv
```

字段至少包括：

```text
station
policy_name
cap_name
season_irrigation_cap
season_n_cap
train_year
train_year_label
eval_year
eval_year_label
cv_type
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

## 14. cap saturation 检查

必须输出：

```text
Leave_One_experiments/season_cap_sensitivity/evaluation/cap_saturation_summary.csv
```

字段至少包括：

```text
station
cap_name
season_irrigation_cap
season_n_cap
train_year
eval_year
total_irrigation
total_n_fertilizer
irrigation_saturation_ratio
n_saturation_ratio
irrigation_at_cap
n_at_cap
num_irrigation_clipped_days
num_n_clipped_days
num_safety_trigger_days
dominant_safety_rule
notes
```

定义：

```text
irrigation_saturation_ratio = total_irrigation / season_irrigation_cap
n_saturation_ratio = total_n_fertilizer / season_n_cap
```

如果 `ratio >= 0.98`，认为基本打满 cap。

---

## 15. 边际收益分析

必须输出：

```text
Leave_One_experiments/season_cap_sensitivity/evaluation/marginal_response_by_cap.csv
```

按站点、cap 排序，计算：

```text
delta_yield_from_previous_cap
delta_reward_from_previous_cap
delta_irrigation_from_previous_cap
delta_n_from_previous_cap
yield_gain_per_100mm_irrigation
yield_gain_per_100kg_n
reward_gain_per_cap_step
```

注意：

FQA/YCA 的结果必须标记：

```text
limited_two_year_cross_validation
```

不要和 HLA/SYA/LCA 完全等价解释。

---

## 16. 图表输出

每个站点至少生成：

```text
{station}_yield_vs_cap.png
{station}_reward_vs_cap.png
{station}_irrigation_n_vs_cap.png
{station}_saturation_ratio_vs_cap.png
{station}_marginal_yield_gain_vs_cap.png
{station}_swfac_nstres_vs_cap.png
```

保存到：

```text
Leave_One_experiments/season_cap_sensitivity/figures/{station}/
```

还需要生成五站点综合图：

```text
all_sites_yield_vs_cap.png
all_sites_reward_vs_cap.png
all_sites_saturation_ratio_vs_cap.png
all_sites_marginal_yield_gain_vs_cap.png
```

保存到：

```text
Leave_One_experiments/season_cap_sensitivity/figures/all_sites/
```

---

## 17. 解释要求

报告中必须明确区分：

```text
action safety cap
```

和：

```text
rainfall-scaling budget scenario
```

解释如下：

1. 当前 cap 敏感性分析是技术诊断，用于判断安全上限是否主导策略；
2. rainfall-scaling budget scenario 是气象情景实验，用于研究不同降雨情景下策略适应性；
3. 两者不是同一个实验；
4. cap 敏感性应该先于正式 budget scenario；
5. 如果 PPO 在所有 cap 下都打满上限，说明 reward 成本项需要加强；
6. 如果出现边际收益递减，可以据此选择更合理的 season cap 作为后续实验设置。

---

## 18. 报告输出

请生成 Markdown 报告：

```text
docs/2026-06-06_season_cap_sensitivity_analysis_report.md
```

报告至少包括：

1. 本阶段目标；
2. 为什么需要 season cap 敏感性分析；
3. cap 与 budget 的区别；
4. 使用的站点、train_year、eval_year；
5. cap 组合；
6. 每个站点运行状态；
7. cap saturation 结果；
8. 产量随 cap 的变化；
9. reward 随 cap 的变化；
10. swfac/nstres 随 cap 的变化；
11. 边际收益分析；
12. 五站点差异；
13. FQA/YCA limited data 的解释；
14. 200/300 是否合理；
15. 是否需要 reward 成本项修正；
16. 下一步建议。

---

## 19. PPT 输出

请生成 PPT：

```text
docs/2026-06-06_season_cap_sensitivity_analysis_report.pptx
```

PPT 至少包括：

1. 任务目标；
2. 为什么要做 cap 敏感性；
3. cap vs budget 区别；
4. 实验设计；
5. HLA pilot 结果；
6. 五站点 cap saturation；
7. 产量响应；
8. reward 响应；
9. 边际收益；
10. 200/300 cap 是否合理；
11. 下一步计划。

并复制一份到：

```text
Leave_One_experiments/season_cap_sensitivity/reports/
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
git add prompts/006_02_season_cap_sensitivity_analysis.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/season_cap_sensitivity/configs/
git add Leave_One_experiments/season_cap_sensitivity/evaluation/
git add Leave_One_experiments/season_cap_sensitivity/figures/
git add Leave_One_experiments/season_cap_sensitivity/reports/
git add docs/2026-06-06_season_cap_sensitivity_analysis_report.md
git add docs/2026-06-06_season_cap_sensitivity_analysis_report.pptx
git commit -m "Run season cap sensitivity analysis for action-safe PPO"
```

注意：

1. 不要默认 commit 大模型 `.zip`；
2. 不要默认 commit tensorboard 大日志；
3. 不要默认 commit 过大的 daily_outputs；
4. 如果需要保存小型模型，请先报告文件大小；
5. 不要强行 push。

---

## 21. 完成后请汇报

完成后请汇报：

1. HLA pilot 是否全部通过；
2. 五站点是否全部完成；
3. 是否所有 cap 都被打满；
4. 哪些站点对 cap 最敏感；
5. 产量是否随 cap 持续升高；
6. reward 是否随 cap 持续升高；
7. 是否出现边际收益递减；
8. 当前 200/300 cap 是否合理；
9. 是否应该进入 reward 成本项修正；
10. 是否应该进入 rainfall-scaling budget scenario；
11. 是否可以做多 seed 稳定性分析。

---

## 22. 最重要原则

1. 本阶段是 cap 敏感性诊断，不是最终管理方案。
2. 不训练无 safety PPO。
3. 不修改 reward。
4. 不做多 seed。
5. 不覆盖旧结果。
6. 不 commit 大模型和大日志。
7. 必须区分 cap 和 budget。
8. 如果所有 cap 都打满，下一步优先 reward 成本项修正，而不是多 seed。
