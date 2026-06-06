# 006_09_train_imitation_learning_prior_from_offline_schedules

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_08_offline_schedule_search_and_bayesian_optimization_prior.md
docs/2026-06-06_offline_schedule_search_report.md
```

现在执行新的子任务：基于 offline schedule search 找到的 expert schedules，训练 imitation learning / behavior cloning prior，为后续 constrained PPO 或 policy initialization 提供可解释先验。

本阶段不要直接回到 PPO multi-seed，不要进入 rainfall-scaling budget scenario，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

上一阶段 `006_08_offline_schedule_search_and_bayesian_optimization_prior` 已经完成。

主要结果：

1. PPO 路线在 006_03 到 006_07 中反复出现 cap saturation。
2. 006_08 暂停 PPO，改用 deterministic offline schedule search。
3. HLA 2011 coarse grid 生成并评估了 598 个 schedules。
4. HLA 找到低投入、产量损失可接受的 expert schedule。
5. HLA best schedule 为：

```text
HLA2011_S0157
mean_yield = 6752.7806
mean_irrigation = 0.0 mm
mean_n = 150.0 kg/ha
mean_yield_loss_vs_ppo_baseline = 0.0179
overall_score = 0.7639
```

6. 该 schedule 通过低投入 <=10% yield-loss gate。
7. 因此扩展到了 SYA/LCA：
   - SYA best schedule: `SYA2012_S0443`, mean_irrigation = 0.0, mean_n = 150.0
   - LCA best schedule: `LCA2010_S0313`, mean_irrigation = 0.0, mean_n = 150.0
8. 已生成 imitation dataset：

```text
Leave_One_experiments/offline_schedule_search/expert_policy/imitation_dataset.csv
```

这说明当前最可靠路线不是继续直接调 PPO，而是先学习 expert schedule 的行为模式。

---

## 2. 本阶段目标

1. 检查 imitation_dataset.csv 是否完整。
2. 基于 expert schedules 构建 supervised imitation learning 数据。
3. 训练一个 behavior cloning policy。
4. 先只做 HLA/SYA/LCA，不做 FQA/YCA。
5. 评估 imitation policy 是否能复现 expert schedules 的低投入策略。
6. 检查 imitation policy 是否不再打满 300/450。
7. 检查产量损失是否可接受。
8. 输出可解释的 imitation policy 结果。
9. 判断是否可以进入 constrained PPO fine-tuning。
10. 判断是否需要先做 expert schedule 数据增强。

---

## 3. 本阶段不要做的事情

1. 不要做 PPO multi-seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练无 action safety 的 PPO。
4. 不要修改 `my_data/` 原始文件。
5. 不要覆盖 006_03 到 006_08 的结果。
6. 不要把 imitation learning 结果直接写成最终 RL 策略。
7. 不要训练 FQA/YCA，除非报告中明确 HLA/SYA/LCA 都通过。
8. 不要忽略 expert schedule 的 deterministic baseline。
9. 不要只报告 imitation loss，必须报告产量、水氮投入、profit、saturation ratio。
10. 不要默认 commit 大模型文件。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_offline_schedule_search_report.md
Leave_One_experiments/offline_schedule_search/expert_policy/imitation_dataset.csv
Leave_One_experiments/offline_schedule_search/expert_policy/HLA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/SYA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/LCA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/evaluation/HLA_top_schedule_cross_year_summary.csv
src/offline_schedule_policy.py
src/run_offline_schedule_search.py
src/ppo_action_safety.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
sb3_wrapper.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/imitation_learning_prior/
```

建议目录结构：

```text
Leave_One_experiments/imitation_learning_prior/
  configs/
  datasets/
  models/
  logs/
  daily_outputs/
  evaluation/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-06_imitation_learning_prior_report.md
docs/2026-06-06_imitation_learning_prior_report.pptx
```

---

## 6. 第一步：检查 imitation dataset

请先生成数据检查报告：

```text
Leave_One_experiments/imitation_learning_prior/evaluation/imitation_dataset_check.md
```

必须检查：

1. 文件是否存在；
2. 包含哪些站点；
3. 包含哪些年份；
4. 总行数；
5. 每个站点/年份行数；
6. 状态变量字段有哪些；
7. expert_action_irrigation 是否存在；
8. expert_action_n 是否存在；
9. action 是否多数为 0；
10. 事件日是否有非零 action；
11. 是否存在 NaN 或 inf；
12. 是否需要标准化；
13. 是否需要按站点分层 split；
14. 是否需要按年份分层 split。

同时输出清洗后的数据：

```text
Leave_One_experiments/imitation_learning_prior/datasets/imitation_dataset_clean.csv
```

---

## 7. 第二步：定义 behavior cloning 任务

本阶段先做 supervised imitation，不做 RL fine-tuning。

建议任务：

```text
输入 X:
  state variables at each day

输出 y:
  expert_action_irrigation
  expert_action_n
```

如果 action 非零样本太少，可以改成两阶段任务：

```text
Task 1: 是否执行灌溉/施肥
Task 2: 如果执行，预测用量
```

优先实现简单、稳定版本：

```text
multi-output regression:
  y = [expert_action_irrigation, expert_action_n]
```

同时可选实现 event classifier：

```text
irrigation_event = expert_action_irrigation > 0
n_event = expert_action_n > 0
```

---

## 8. 第三步：模型选择

先不要上复杂深度 RL。优先训练简单可解释模型。

至少训练以下模型：

```text
BC_constant_schedule_baseline
BC_random_forest_regressor
BC_mlp_regressor
```

可选：

```text
BC_two_stage_classifier_regressor
```

说明：

1. constant schedule baseline 直接复现 best expert schedule；
2. random forest 作为强基线；
3. MLP regressor 用于后续 policy initialization 的可能性；
4. 如果 action 极度稀疏，两阶段模型可能更合理。

---

## 9. 第四步：训练/验证 split

不要随机打乱导致同一年数据泄漏。

建议 split：

```text
train:
  HLA train_year 2011
  SYA train_year 2012
  LCA train_year 2010

validation:
  HLA 2007, 2009
  SYA 2014, 2015
  LCA 2008, 2009, 2011
```

如果 imitation_dataset.csv 只包含 expert schedule daily states 而不包含所有验证年份，请用 expert schedule 重新生成对应年份 daily dataset。

输出 split 文件：

```text
Leave_One_experiments/imitation_learning_prior/datasets/train_split.csv
Leave_One_experiments/imitation_learning_prior/datasets/validation_split.csv
```

---

## 10. 第五步：训练 behavior cloning models

新增或更新：

```text
src/train_imitation_policy.py
src/imitation_policy_models.py
src/evaluate_imitation_policy.py
```

训练输出：

```text
Leave_One_experiments/imitation_learning_prior/models/
```

评价指标包括：

```text
irrigation_mae
nitrogen_mae
irrigation_rmse
nitrogen_rmse
irrigation_event_precision
irrigation_event_recall
n_event_precision
n_event_recall
nonzero_action_accuracy
zero_action_accuracy
```

保存：

```text
Leave_One_experiments/imitation_learning_prior/evaluation/imitation_supervised_metrics.csv
```

---

## 11. 第六步：把 imitation policy 放回 gym-DSSAT 评估

只看 supervised loss 不够，必须在 DSSAT/gym-DSSAT 中评估 imitation policy 的实际效果。

对每个 imitation policy，在以下年份评估：

```text
HLA: 2011, 2007, 2009
SYA: 2012, 2014, 2015
LCA: 2010, 2011, 2008, 2009
```

保存 daily CSV 到：

```text
Leave_One_experiments/imitation_learning_prior/daily_outputs/
```

summary 保存到：

```text
Leave_One_experiments/imitation_learning_prior/evaluation/imitation_policy_dssat_evaluation_summary.csv
```

字段至少包括：

```text
station
policy_name
model_type
eval_year
run_status
episode_completed
final_grnwt
total_irrigation
total_n_fertilizer
mean_swfac
mean_nstres
profit_score
yield_loss_vs_expert_schedule
yield_loss_vs_ppo_baseline
irrigation_diff_vs_expert
n_diff_vs_expert
daily_csv_path
notes
```

---

## 12. 第七步：与 expert schedule 和 PPO baseline 比较

必须比较三类策略：

```text
expert_schedule
imitation_policy
ppo_cap_saturated_baseline
```

输出：

```text
Leave_One_experiments/imitation_learning_prior/evaluation/policy_comparison_expert_bc_ppo.csv
```

字段至少包括：

```text
station
year
policy_type
policy_name
final_grnwt
total_irrigation
total_n_fertilizer
profit_score
mean_swfac
mean_nstres
yield_loss_vs_ppo
input_reduction_vs_ppo
notes
```

重点判断：

1. imitation policy 是否接近 expert schedule；
2. imitation policy 是否避免 300/450 饱和；
3. imitation policy 相比 PPO baseline 是否显著节水/节氮；
4. imitation policy 的产量损失是否可接受；
5. imitation policy 是否跨年份稳定。

---

## 13. 第八步：是否可以进入 constrained PPO fine-tuning

如果 imitation policy 满足：

```text
mean_irrigation <= 100 mm
mean_n <= 200 kg/ha
mean_yield_loss_vs_ppo_baseline <= 15%
run_status ok for all evaluated years
```

则报告建议进入下一阶段：

```text
006_10_constrained_ppo_finetuning_from_imitation_prior
```

否则建议：

```text
继续改 expert dataset 或 imitation model
```

不要直接进入 PPO multi-seed。

---

## 14. 图表输出

至少生成：

```text
imitation_action_distribution.png
expert_vs_imitation_daily_actions.png
expert_vs_imitation_cumulative_inputs.png
policy_comparison_yield_vs_input.png
policy_comparison_profit.png
policy_comparison_by_site.png
imitation_generalization_by_year.png
```

保存到：

```text
Leave_One_experiments/imitation_learning_prior/figures/
```

---

## 15. 报告要求

报告必须说明：

1. 为什么现在转向 imitation learning；
2. expert schedules 来自哪里；
3. imitation dataset 的组成；
4. BC 模型设计；
5. supervised metrics；
6. DSSAT/gym-DSSAT 实际评估结果；
7. imitation policy 是否避免 cap saturation；
8. 与 expert schedule 和 PPO baseline 的对比；
9. 是否可以进入 constrained PPO fine-tuning；
10. 是否仍需要 offline search 数据增强；
11. 是否可以做 rainfall-scaling budget scenario。

---

## 16. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_imitation_learning_prior_report.md
```

生成 PPT：

```text
docs/2026-06-06_imitation_learning_prior_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/imitation_learning_prior/reports/
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
git add prompts/006_09_train_imitation_learning_prior_from_offline_schedules.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/imitation_learning_prior/configs/
git add Leave_One_experiments/imitation_learning_prior/datasets/
git add Leave_One_experiments/imitation_learning_prior/evaluation/
git add Leave_One_experiments/imitation_learning_prior/figures/
git add Leave_One_experiments/imitation_learning_prior/reports/
git add docs/2026-06-06_imitation_learning_prior_report.md
git add docs/2026-06-06_imitation_learning_prior_report.pptx
git commit -m "Train imitation learning prior from offline expert schedules"
```

注意：

1. 不要默认 commit 大模型；
2. 不要默认 commit 大量 daily_outputs；
3. 不要默认 commit tensorboard；
4. 不要强行 push。

---

## 18. 完成后请汇报

完成后请汇报：

1. imitation_dataset.csv 是否完整；
2. 训练了哪些 BC models；
3. supervised metrics 如何；
4. DSSAT/gym-DSSAT 评估是否通过；
5. imitation policy 是否避免 300/450 饱和；
6. imitation policy 与 expert schedule 的差距；
7. imitation policy 与 PPO saturated baseline 的差距；
8. 是否可以进入 constrained PPO fine-tuning；
9. 是否需要 expert dataset augmentation；
10. 是否可以进入 rainfall-scaling budget scenario。
