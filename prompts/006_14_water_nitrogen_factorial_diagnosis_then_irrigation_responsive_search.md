# 006_14_water_nitrogen_factorial_diagnosis_then_irrigation_responsive_search

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_12_expert_dataset_augmentation_before_constrained_ppo.md
prompts/006_13_multiseed_constrained_ppo_with_augmented_prior.md
docs/2026-06-06_expert_dataset_augmentation_report.md
docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.md
docs/experiment_records/all_water_stress_summary_final.md
```

如果文件名或日期略有不同，请搜索关键词：

```text
expert_dataset_augmentation_report
constrained_ppo_multiseed_augmented_prior_report
all_water_stress_summary_final
BC_random_forest_regressor_augmented
MS0_augmented_RF_prior_replay
swfac
water nitrogen factorial
```

现在执行新的子任务：**先不要继续训练 PPO，也不要直接进入 rainfall-scaling。先做 water × nitrogen factorial diagnosis，确认真实 observed-year 环境下灌水到底有没有用；只有证明灌水有价值，才继续做 irrigation-responsive expert schedule search。**

---

## 1. 本阶段一句话目标

本阶段要回答一个非常具体的问题：

```text
在当前五个站点的实测年份中，产量提升主要来自氮，还是来自水？
这些站点年份到底有没有真实灌溉优化价值？
```

如果结论是：

```text
同等施氮下，加灌溉不能提高产量 / profit，也不能明显降低 swfac 水分胁迫
```

那就不要继续把灌溉作为 observed-year 最优策略来训练 PPO。

如果结论是：

```text
某些站点-年份在不灌水时有 swfac 水分胁迫；
加灌溉后 swfac 降低；
同等施氮下产量或 profit 明显提高
```

那才继续为这些站点-年份构建 irrigation-responsive expert schedules。

---

## 2. 本阶段背景

前面阶段的主要结果如下：

### 006_12

expert dataset augmentation 成功，把 expert schedules 从 3 个扩展到 45 个：

```text
HLA: 15
LCA: 15
SYA: 15
```

并训练出推荐 prior：

```text
BC_random_forest_regressor_augmented
mean_yield = 6503.2767
mean_irrigation = 0.0
mean_n = 81.8291
mean_profit = 44.5755
mean_yield_loss_vs_ppo = 0.0542
```

但是灌溉样本仍然很少：

```text
00609_original:
  rows = 1300
  nonzero_irrigation_rows = 0

00612_augmented:
  rows = 19500
  nonzero_irrigation_rows = 87
```

也就是说，BC/RF prior 很容易学成：

```text
几乎永远不灌水
```

### 006_13

constrained PPO multi-seed with augmented RF prior 已完成。

结果显示：

```text
MS0_augmented_RF_prior_replay:
  mean_yield = 6500.5829
  mean_profit = 44.259
  mean_irrigation = 0.0
  mean_n = 82.9872
  cap_regression_rate = 0.0

MS1_residual_augmented_RF_prior_strict:
  mean_yield = 6462.8087
  mean_profit = 34.2437
  mean_irrigation = 17.2507
  mean_n = 87.0362
  cap_regression_rate = 0.0

MS2_guardrail_augmented_RF_100_200:
  mean_yield = 8498.0336
  mean_profit = -15.0197
  mean_irrigation = 100.0
  mean_n = 200.0
  cap_regression_rate = 0.0
```

结论是：

```text
PPO 没有退回 300/450；
但 PPO fine-tuning 没有优于 augmented RF prior；
灌溉仍然没有表现出稳定经济收益；
rainfall-scaling 仍然过早。
```

### all-mode 水分胁迫诊断

已有 all-mode 诊断显示：

```text
HLA: PRCP - ETCP = -150.4 mm，但 swfac_stress_days_gt_0.05 = 0
FQ: swfac_stress_days_gt_0.05 = 0
YC: swfac_stress_days_gt_0.05 = 0
SY/LC: 部分 all-mode 诊断 missing_or_stalled，需要补跑或谨慎处理
```

这说明：

```text
当前 observed years 里，还没有稳定证据表明某个站点存在强水分胁迫。
```

因此，本阶段不要直接做 rainfall-scaling，也不要直接继续 PPO。先把水和氮的贡献拆开。

---

## 3. 本阶段不要做的事情

1. 不要训练普通 PPO。
2. 不要训练无 action safety PPO。
3. 不要继续 constrained PPO multi-seed。
4. 不要进入 rainfall-scaling budget scenario。
5. 不要直接把 irrigation=0 解释成最终农学结论。
6. 不要直接手工往 imitation CSV 里加灌溉行。
7. 不要只看最高产量，必须同时看 water、nitrogen、profit、swfac、nstres。
8. 不要覆盖 006_08 到 006_13 的结果。
9. 不要修改 `my_data/` 原始文件。
10. 不要默认 commit 大量 daily_outputs、模型文件或 tensorboard。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.md
docs/2026-06-06_expert_dataset_augmentation_report.md
docs/experiment_records/all_water_stress_summary_final.md

Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/multiseed_constrained_ppo_summary.csv
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/multiseed_method_comparison.csv
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/policy_comparison_multiseed_augmented_prior.csv

Leave_One_experiments/expert_dataset_augmentation/evaluation/augmented_imitation_policy_dssat_summary.csv
Leave_One_experiments/expert_dataset_augmentation/evaluation/recommended_augmented_prior_policy.csv
Leave_One_experiments/expert_dataset_augmentation/evaluation/augmented_dataset_action_distribution_summary.csv

src/offline_schedule_policy.py
src/run_offline_schedule_search.py
src/ppo_safe_rendering.py
src/ppo_action_safety.py
src/replay_imitation_prior.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/
```

建议目录结构：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/
  configs/
  fixed_factorial_schedules/
  daily_outputs/
  evaluation/
  irrigation_responsive_search/
  expert_policy/
  imitation_dataset/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-06_water_nitrogen_factorial_diagnosis_report.md
docs/2026-06-06_water_nitrogen_factorial_diagnosis_report.pptx
```

---

## 6. 第一步：站点级现状诊断

请先生成：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/current_site_status_review.md
```

必须说明：

1. 006_13 中 MS0/MS1/MS2 的总体结果；
2. 为什么不继续 PPO；
3. 为什么不直接 rainfall-scaling；
4. 当前 imitation prior 为什么容易学成不灌水；
5. all-mode 水分胁迫诊断里哪些站点有 swfac，哪些没有；
6. 哪些站点 all-mode 诊断 missing_or_stalled，需要补充说明；
7. 当前最需要回答的问题是“灌水有没有真实收益”。

同时输出站点级汇总表：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/current_site_status_summary.csv
```

字段至少包括：

```text
station
observed_years
PRCP
ETCP
PRCP_minus_ETCP
all_mode_swfac_available
swfac_stress_days_gt_0p05
max_swfac
mean_swfac
nstres_days_gt_0p05
max_nstres
mean_nstres
augmented_rf_mean_yield
augmented_rf_mean_irrigation
augmented_rf_mean_n
augmented_rf_mean_profit
diagnosis_note
```

---

## 7. 第二步：water × nitrogen factorial diagnosis

这是本阶段最重要的部分。

请不要直接搜索最优 schedule。先做固定因子化对照，拆开水和氮的贡献。

### 7.1 站点和年份

优先对五个站点的 observed years 做诊断：

```text
HLA: 2007, 2009, 2011
SYA: 2012, 2014, 2015
LCA: 2008, 2009, 2010, 2011
FQA: 2008, 2010
YCA: 2008, 2014
```

如果 FQA/YCA 的当前配置不完整，请报告原因，不要强行跳过不说明。

### 7.2 固定方案

每个站点-年份至少跑以下 fixed schedules：

```text
T0_null_zero:
  total_irrigation = 0
  total_n = 0

T1_N_only_low:
  irrigation = 0
  N total = 75 kg/ha
  N events = DAP 1, 30, 60

T2_N_only_medium:
  irrigation = 0
  N total = 150 kg/ha
  N events = DAP 1, 30, 60

T3_I_only_low:
  irrigation total = 60 mm
  irrigation events = DAP 35, 65
  N total = 0

T4_N_medium_I_low:
  irrigation total = 60 mm
  irrigation events = DAP 35, 65
  N total = 150 kg/ha
  N events = DAP 1, 30, 60

T5_N_medium_I_mid:
  irrigation total = 120 mm
  irrigation events = DAP 25, 50, 75
  N total = 150 kg/ha
  N events = DAP 1, 30, 60

T6_N_high_I_mid:
  irrigation total = 120 mm
  irrigation events = DAP 25, 50, 75
  N total = 225 kg/ha
  N events = DAP 1, 30, 60
```

说明：

```text
T1/T2 用来测氮效应；
T3 用来测纯灌溉效应；
T4/T5 用来测同等 N 下灌溉是否增产；
T6 用来测水氮共同高投入是否只是氮效应或水氮互作。
```

### 7.3 输出

新增或更新：

```text
src/run_water_nitrogen_factorial_diagnosis.py
```

每个 fixed schedule 保存 daily CSV 和 summary。

summary 输出：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/water_nitrogen_factorial_summary.csv
```

字段至少包括：

```text
station
year
treatment_id
treatment_name
total_irrigation
total_n
run_status
episode_completed
final_grnwt
final_topwt
final_xlai
profit_default
profit_low_water_cost
mean_swfac
max_swfac
swfac_stress_days_gt_0p05
swfac_stress_days_gt_0p10
mean_nstres
max_nstres
nstres_days_gt_0p05
yield_gain_vs_T0
yield_gain_vs_N_only_medium
yield_gain_from_irrigation_at_same_N
yield_gain_from_n_at_same_irrigation
profit_gain_from_irrigation_at_same_N_default
profit_gain_from_irrigation_at_same_N_low_water_cost
water_productivity
n_productivity
daily_csv_path
notes
```

profit 计算：

```text
profit_default = 0.01 * final_grnwt - 0.5 * total_irrigation - 0.25 * total_n
profit_low_water_cost = 0.01 * final_grnwt - 0.1 * total_irrigation - 0.25 * total_n
```

---

## 8. 第三步：判断哪些站点-年份有灌溉响应

基于 factorial summary，生成：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/irrigation_response_screening.csv
```

每个站点-年份判断：

```text
has_water_stress
irrigation_reduces_swfac
irrigation_increases_yield_at_same_N
irrigation_increases_profit_default
irrigation_increases_profit_low_water_cost
irrigation_candidate
```

建议判定标准：

```text
has_water_stress:
  T2_N_only_medium 中 swfac_stress_days_gt_0p05 > 0
  或 max_swfac > 0.05

irrigation_reduces_swfac:
  T4_N_medium_I_low 或 T5_N_medium_I_mid 的 max_swfac / stress days 低于 T2

irrigation_increases_yield_at_same_N:
  T4 或 T5 final_grnwt > T2 final_grnwt * 1.03
  或 yield gain >= 200 kg/ha

irrigation_increases_profit_default:
  T4 或 T5 profit_default > T2 profit_default

irrigation_increases_profit_low_water_cost:
  T4 或 T5 profit_low_water_cost > T2 profit_low_water_cost

irrigation_candidate:
  irrigation_reduces_swfac 为 True
  且 irrigation_increases_yield_at_same_N 为 True
  且至少在 low_water_cost 下 profit 不差
```

同时输出文字报告：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/irrigation_response_screening_report.md
```

必须回答：

1. 哪些站点-年份有水分胁迫；
2. 哪些站点-年份加灌溉能降低 swfac；
3. 哪些站点-年份同等 N 下加灌溉能增产；
4. 哪些站点-年份灌溉在 default water cost 下经济；
5. 哪些只在 low water cost 下经济；
6. 哪些站点-年份完全不适合灌溉优化；
7. HLA yield gap 是否真的是灌溉问题；
8. FQA 2008 是否比 HLA 更像 observed-year 灌溉候选。

---

## 9. 第四步：只对候选站点-年份做 irrigation-responsive expert search

只有当第 8 节筛选出 `irrigation_candidate = True` 的站点-年份，才继续做 irrigation-responsive expert search。

如果没有任何站点-年份通过 screening，不要强行搜索。直接生成 negative conclusion：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/no_observed_year_irrigation_response_conclusion.md
```

内容说明：

```text
当前 observed years 下，灌溉不是主要优化维度；
后续 rainfall-scaling 只能作为人为干旱 stress test；
当前主线应保留 augmented RF / nitrogen-management prior。
```

### 9.1 搜索空间

对筛选出的候选站点-年份，使用：

```text
I events:
  DAP 20
  DAP 35
  DAP 50
  DAP 65
  DAP 80
  DAP 95

I amount each event:
  [0, 20, 40, 60]

N events:
  DAP 1
  DAP 30
  DAP 60

N amount each event:
  [0, 50, 75, 100, 150]
```

约束：

```text
total_irrigation <= 160 mm
total_n <= 250 kg/ha
```

如果组合过多，采用分阶段搜索：

```text
Stage A: 固定几个较优 N schedule，只搜索 irrigation
Stage B: 固定 top irrigation schedule，再搜索 N
Stage C: 对 Pareto schedules 局部 refine
```

### 9.2 输出

新增或更新：

```text
src/run_irrigation_responsive_search.py
```

输出：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/irrigation_responsive_search_summary.csv
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/irrigation_responsive_pareto.csv
Leave_One_experiments/water_nitrogen_factorial_diagnosis/expert_policy/irrigation_responsive_expert_schedule_ranking.csv
```

字段至少包括：

```text
station
year
schedule_id
final_grnwt
total_irrigation
total_n
profit_default
profit_low_water_cost
mean_swfac
max_swfac
swfac_stress_days_gt_0p05
mean_nstres
max_nstres
yield_gain_vs_augmented_rf_prior
profit_gain_vs_augmented_rf_prior_default
profit_gain_vs_augmented_rf_prior_low_water_cost
yield_per_100mm_irrigation
recommendation_type
is_pareto
daily_csv_path
notes
```

---

## 10. 第五步：是否更新 expert library / imitation dataset

只有当 irrigation-responsive search 找到有效灌溉 schedules，才更新 expert library 和 imitation dataset。

有效条件：

```text
total_irrigation > 0
yield_gain_vs_augmented_rf_prior >= 0 或 yield_loss <= 10%
profit_low_water_cost 不低于 augmented RF prior
swfac 有改善或 yield 有改善
run_status ok
```

如果有效，输出：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/expert_policy/irrigation_responsive_expert_library.csv
Leave_One_experiments/water_nitrogen_factorial_diagnosis/imitation_dataset/imitation_dataset_irrigation_responsive.csv
```

如果无效，不要强行生成新的 imitation dataset。

---

## 11. 第六步：是否重新训练 irrigation-responsive BC

只有当：

```text
irrigation_positive_schedules >= 5
nonzero_irrigation_rows 明显高于 006_12
unique_irrigation_amounts >= 3
```

才重新训练：

```text
BC_random_forest_regressor_irrigation_responsive
BC_two_stage_classifier_regressor_irrigation_responsive
```

否则不要训练新 BC，只报告样本不足。

---

## 12. 第七步：是否建议 rainfall-scaling

本阶段结束后，按以下规则判断：

### 情况 A：observed years 中找到稳定灌溉响应

如果：

```text
有多个站点-年份 irrigation_candidate=True
irrigation-positive schedules 跨年稳定
imitation policy 能学到非零灌溉
profit_low_water_cost 或合理经济参数下不劣于 irrigation=0 prior
```

则可以建议后续设计 rainfall-scaling stress test，但仍要说明：

```text
rainfall-scaling 是压力测试，不是 observed-year 直接结论。
```

### 情况 B：observed years 没有稳定灌溉响应

如果：

```text
没有站点-年份通过 irrigation_response_screening
```

则明确建议：

```text
不要进入 rainfall-scaling 作为主线；
当前 observed-year 主线应聚焦氮管理 / expert-imitation policy；
rainfall-scaling 只能另立为人为干旱 stress test。
```

---

## 13. 图表输出

至少生成：

```text
factorial_yield_by_treatment_site_year.png
factorial_profit_by_treatment_site_year.png
factorial_swfac_by_treatment_site_year.png
yield_gain_from_irrigation_at_same_N.png
profit_gain_from_irrigation_at_same_N.png
irrigation_response_screening_heatmap.png
```

如果执行了 irrigation-responsive search，再生成：

```text
irrigation_responsive_yield_response.png
irrigation_responsive_profit_response_default_cost.png
irrigation_responsive_profit_response_low_water_cost.png
irrigation_responsive_pareto_frontier.png
irrigation_timing_effect.png
```

保存到：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/figures/
```

---

## 14. 报告要求

报告必须说明：

1. 为什么 006_13 后不能直接 rainfall-scaling；
2. 为什么要先做 water × nitrogen factorial diagnosis；
3. 各站点-年份水氮拆分结果；
4. 产量提升主要来自水还是氮；
5. 哪些站点-年份存在 swfac 水分胁迫；
6. 同等 N 下加灌溉是否增产；
7. 灌溉在 default water cost 和 low water cost 下是否经济；
8. HLA yield gap 是否是灌溉问题；
9. FQA 2008 是否更像灌溉候选；
10. 是否执行 irrigation-responsive expert search；
11. 是否更新 expert library / imitation dataset；
12. 是否重新训练 BC；
13. 是否建议 rainfall-scaling；
14. 如果不建议，下一步是什么。

---

## 15. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_water_nitrogen_factorial_diagnosis_report.md
```

生成 PPT：

```text
docs/2026-06-06_water_nitrogen_factorial_diagnosis_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/water_nitrogen_factorial_diagnosis/reports/
```

---

## 16. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/006_14_water_nitrogen_factorial_diagnosis_then_irrigation_responsive_search.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/water_nitrogen_factorial_diagnosis/configs/
git add Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/
git add Leave_One_experiments/water_nitrogen_factorial_diagnosis/expert_policy/
git add Leave_One_experiments/water_nitrogen_factorial_diagnosis/imitation_dataset/
git add Leave_One_experiments/water_nitrogen_factorial_diagnosis/figures/
git add Leave_One_experiments/water_nitrogen_factorial_diagnosis/reports/
git add docs/2026-06-06_water_nitrogen_factorial_diagnosis_report.md
git add docs/2026-06-06_water_nitrogen_factorial_diagnosis_report.pptx
git commit -m "Diagnose water nitrogen factorial response before rainfall scaling"
```

注意：

1. 不要默认 commit 大量 daily_outputs；
2. 不要默认 commit 大模型；
3. 不要默认 commit tensorboard；
4. 不要强行 push。

---

## 17. 完成后请汇报

完成后请汇报：

1. 哪些站点-年份完成了 water × nitrogen factorial diagnosis；
2. 产量提升主要来自水还是氮；
3. 哪些站点-年份有 swfac 水分胁迫；
4. 哪些站点-年份同等 N 下加灌溉能增产；
5. 哪些站点-年份灌溉在 default water cost 下经济；
6. 哪些只在 low water cost 下经济；
7. 是否执行 irrigation-responsive expert search；
8. 是否更新 expert library / imitation dataset；
9. 是否重新训练 BC；
10. 是否建议 rainfall-scaling；
11. 如果不建议，下一步是什么。
