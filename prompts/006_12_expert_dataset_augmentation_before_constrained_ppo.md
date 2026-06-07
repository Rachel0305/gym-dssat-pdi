# 006_12_expert_dataset_augmentation_before_constrained_ppo

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_08_offline_schedule_search_and_bayesian_optimization_prior.md
prompts/006_09_train_imitation_learning_prior_from_offline_schedules.md
prompts/006_10_constrained_ppo_finetuning_from_imitation_prior.md
prompts/006_11_debug_prior_replay_and_constrained_finetuning_consistency.md
docs/2026-06-06_offline_schedule_search_report.md
docs/2026-06-06_imitation_learning_prior_report.md
docs/2026-06-06_constrained_ppo_finetuning_report.md
docs/2026-06-06_prior_replay_debug_report.md
```

如果文件名或日期略有不同，请搜索关键词：

```text
offline_schedule_search_report
imitation_learning_prior_report
constrained_ppo_finetuning_report
prior_replay_debug_report
BC_two_stage_classifier_regressor
FT0_BC_two_stage_replay_fixed
```

现在执行新的子任务：在确认 prior replay 已修复后，不继续盲目 fine-tune PPO，而是扩充 expert dataset，增强 imitation prior 的覆盖范围，为下一轮 constrained PPO 或 rainfall-scaling 前的稳定验证做准备。

本阶段不要做 PPO multi-seed，不要进入 rainfall-scaling budget scenario，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

上一阶段 `006_11_debug_prior_replay_and_constrained_finetuning_consistency` 已经完成。

关键结果：

### 1.1 mismatch 已查明

`006_10` 的 FT0 没有复现 `006_09`，原因是 action safety 接口不一致：

1. `SafeActionWrapper` 读取 environment `dap`，早期 rows 里 dap=0，导致 first-day N events 被 `anfer_dap_range` 拒绝；
2. `006_10` 使用 `daily_n_max = 80`，裁剪了 HLA/LCA 的 BC prior N events；
3. `006_09` replay 使用 exported BC action tables、sim-day action safety 和 daily N max 150。

### 1.2 pure replay 已修复

修复后 pure replay 结果与 `006_09` 一致：

```text
mean_yield = 8091.8537
mean_n = 153.575
mean_irrigation = 0.0
yield_loss_vs_00609_bc_two_stage = 0.0
n_diff_vs_00609_bc_two_stage = 0.0
```

### 1.3 修复后的 constrained PPO 结果

```text
FT0_BC_two_stage_replay_fixed:
  mean_yield = 8091.8537
  mean_profit = 42.5248
  mean_i = 0.0
  mean_n = 153.575
  mean_loss_ppo = -0.1768
  mean_loss_bc = 0.0
  profit_gap_vs_bc_two_stage = 0.0

FT2_residual_bc_prior_fixed:
  mean_yield = 6874.2549
  mean_profit = -1.1707
  mean_i = 71.192
  mean_n = 137.2691
  mean_loss_ppo = 0.0002
  mean_loss_bc = 0.2089
  profit_gap_vs_bc_two_stage = -43.6955

FT3_budget_guardrail_100_200_fixed:
  mean_yield = 8460.3945
  mean_profit = -14.8825
  mean_i = 98.973
  mean_n = 200.0
  mean_loss_ppo = -0.2304
  mean_loss_bc = -0.0453
  profit_gap_vs_bc_two_stage = -57.4073
```

### 1.4 当前推荐

报告推荐：

```text
FT0_BC_two_stage_replay_fixed
```

也就是说，当前最稳的策略仍是 BC prior / expert replay，不是 fine-tuned PPO。

因此，本阶段不要进入 PPO multi-seed。下一步应该扩充 expert dataset，使 learned imitation policy 覆盖更多 schedule、更多水氮组合和更多站点/年份，然后再判断是否回到 constrained PPO。

---

## 2. 本阶段核心判断

当前结论是：

1. prior replay consistency 已修复；
2. BC_two_stage prior 是可靠的；
3. 但 constrained PPO fine-tuning 没有超过 BC prior；
4. FT2/FT3 虽然避免了 300/450 饱和，但 profit 明显低于 BC prior；
5. 直接做 multi-seed constrained PPO 没意义；
6. 需要先扩充 expert dataset，让 imitation prior 不只是学习少数 best schedules；
7. 之后再考虑 constrained PPO 或 rainfall-scaling。

---

## 3. 本阶段目标

本阶段目标是：

1. 扩充 offline expert schedules；
2. 为 HLA/SYA/LCA 生成更多 Pareto-balanced expert trajectories；
3. 可选补充 FQA/YCA 的 two-year expert schedules；
4. 重新构建 richer imitation dataset；
5. 训练更稳健的 imitation prior；
6. 检查 learned policy 是否仍保持低投入和高产；
7. 决定是否进入 constrained PPO multi-seed 或 rainfall-scaling。

---

## 4. 本阶段不要做的事情

1. 不要做 PPO multi-seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练无 action safety PPO。
4. 不要直接重跑 006_10 的全部 fine-tuning。
5. 不要覆盖 006_08 到 006_11 的结果。
6. 不要修改 `my_data/` 原始文件。
7. 不要只使用单个 best schedule 做 imitation；
8. 不要忽略 low-input / medium-input / high-yield 的多样性；
9. 不要把 offline expert schedules 直接写成最终 RL 策略。

---

## 5. 输入文件

优先读取：

```text
docs/2026-06-06_prior_replay_debug_report.md
docs/2026-06-06_offline_schedule_search_report.md
docs/2026-06-06_imitation_learning_prior_report.md

Leave_One_experiments/prior_replay_debug/evaluation/replayed_bc_two_stage_evaluation_summary.csv
Leave_One_experiments/prior_replay_debug/evaluation/constrained_ppo_fixed_interface_summary.csv

Leave_One_experiments/offline_schedule_search/expert_policy/HLA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/SYA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/LCA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/imitation_dataset.csv

Leave_One_experiments/imitation_learning_prior/datasets/imitation_dataset_clean.csv
Leave_One_experiments/imitation_learning_prior/evaluation/imitation_policy_dssat_evaluation_summary.csv

src/offline_schedule_policy.py
src/run_offline_schedule_search.py
src/train_imitation_policy.py
src/imitation_policy_models.py
src/evaluate_imitation_policy.py
src/replay_imitation_prior.py
src/ppo_action_safety.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 6. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/expert_dataset_augmentation/
```

建议目录结构：

```text
Leave_One_experiments/expert_dataset_augmentation/
  configs/
  candidate_schedules/
  daily_outputs/
  evaluation/
  expert_policy/
  imitation_dataset/
  models/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-06_expert_dataset_augmentation_report.md
docs/2026-06-06_expert_dataset_augmentation_report.pptx
```

---

## 7. 第一步：复盘为什么不进入 PPO multi-seed

请生成：

```text
Leave_One_experiments/expert_dataset_augmentation/evaluation/why_not_multiseed_yet.md
```

内容至少包括：

1. 006_11 修复了什么；
2. 为什么 FT0 已经复现 BC prior；
3. 为什么 FT2/FT3 不推荐；
4. 为什么此时做 PPO multi-seed 没意义；
5. 为什么需要 richer expert dataset；
6. richer expert dataset 如何支持后续 constrained PPO 或 rainfall-scaling。

---

## 8. 第二步：扩充 expert schedules 的选择原则

不要只保留单个 best schedule。每个站点至少保留以下类型：

```text
top_profit
top_yield
pareto_balanced
low_input_within_5pct_yield_loss
low_input_within_10pct_yield_loss
medium_input_stable
```

每个类型最多保留 5 个 schedule，去重后每站点建议保留 10-25 个 expert schedules。

筛选标准建议：

```text
run_status ok
episode_completed true
mean_yield_loss_vs_ppo_baseline <= 15%
mean_n <= 300 kg/ha
mean_irrigation <= 250 mm
not dominated on yield-input-profit Pareto frontier
```

输出：

```text
Leave_One_experiments/expert_dataset_augmentation/expert_policy/{station}_augmented_expert_schedule_list.csv
```

字段至少包括：

```text
station
schedule_id
expert_type
mean_yield
std_yield
mean_profit
mean_irrigation
mean_n
mean_yield_loss_vs_ppo_baseline
overall_score
selected_reason
```

---

## 9. 第三步：重新生成 expert trajectories

对每个入选 schedule，在对应站点所有 observed eval years 上重新运行 deterministic schedule policy，生成 daily trajectories。

站点和年份：

```text
HLA:
  years = 2011, 2007, 2009

SYA:
  years = 2012, 2014, 2015

LCA:
  years = 2010, 2011, 2008, 2009
```

如果时间允许，可选扩展：

```text
FQA:
  years = 2010, 2008

YCA:
  years = 2008, 2014
```

FQA/YCA 只能标记为：

```text
limited_two_year_expert_dataset
```

daily output 保存到：

```text
Leave_One_experiments/expert_dataset_augmentation/daily_outputs/
```

summary 保存到：

```text
Leave_One_experiments/expert_dataset_augmentation/evaluation/augmented_expert_schedule_evaluation_summary.csv
```

---

## 10. 第四步：构建 richer imitation dataset

把所有 selected expert trajectories 合并成：

```text
Leave_One_experiments/expert_dataset_augmentation/imitation_dataset/imitation_dataset_augmented.csv
```

字段至少包括：

```text
station
year
schedule_id
expert_type
date
dap
state_variables
expert_action_irrigation
expert_action_n
total_schedule_irrigation
total_schedule_n
profit_score
yield_rank_group
input_level_group
```

必须检查：

1. 每个站点多少 schedules；
2. 每个站点多少 daily rows；
3. 非零 N action 占比；
4. 非零 irrigation action 占比；
5. action 分布是否比 006_09 更丰富；
6. 是否存在 NaN/inf；
7. 是否存在单位不一致；
8. 是否需要标准化。

生成检查报告：

```text
Leave_One_experiments/expert_dataset_augmentation/evaluation/augmented_imitation_dataset_check.md
```

---

## 11. 第五步：重新训练 imitation models

在 augmented dataset 上重新训练：

```text
BC_random_forest_regressor_augmented
BC_two_stage_classifier_regressor_augmented
BC_mlp_regressor_augmented
```

可选：

```text
BC_transformer_or_sequence_model_augmented
```

但可选模型不要影响主流程。

输出：

```text
Leave_One_experiments/expert_dataset_augmentation/models/
Leave_One_experiments/expert_dataset_augmentation/evaluation/augmented_imitation_supervised_metrics.csv
```

训练/验证 split 原则：

1. 不要随机泄漏同一 schedule 的相邻日；
2. 优先按 year 或 schedule 分组 split；
3. 至少报告 in-site validation 和 cross-year validation；
4. 如果包含 FQA/YCA，单独标记 limited two-year validation。

---

## 12. 第六步：在 DSSAT/gym-DSSAT 中评估 augmented imitation policies

对以下 policy 进行 deterministic evaluation：

```text
BC_two_stage_classifier_regressor_original
BC_two_stage_classifier_regressor_augmented
BC_random_forest_regressor_augmented
BC_mlp_regressor_augmented
best_expert_schedule_replay
old_ppo_cap_saturated_baseline
```

站点：

```text
HLA, SYA, LCA
```

可选：

```text
FQA, YCA
```

输出：

```text
Leave_One_experiments/expert_dataset_augmentation/evaluation/augmented_imitation_policy_dssat_summary.csv
```

字段至少包括：

```text
station
policy_name
eval_year
run_status
episode_completed
final_grnwt
total_irrigation
total_n_fertilizer
profit_score
mean_swfac
mean_nstres
yield_loss_vs_expert_best
yield_loss_vs_ppo_baseline
input_reduction_vs_ppo_baseline
daily_csv_path
notes
```

---

## 13. 第七步：选择下一阶段 policy

选择推荐 policy 的标准：

```text
run_status ok for all eval years
mean_irrigation <= 100 mm 或显著低于 PPO baseline
mean_n <= 200 kg/ha 或显著低于 PPO baseline
mean_yield_loss_vs_ppo_baseline <= 15%
profit_score 不低于 original BC_two_stage prior 的 90%
action 分布比 original prior 更丰富
not saturated at 300/450
```

输出：

```text
Leave_One_experiments/expert_dataset_augmentation/evaluation/recommended_augmented_prior_policy.csv
```

如果没有 learned policy 优于 original BC_two_stage，则保留 original BC_two_stage / expert replay，不进入 PPO fine-tuning。

---

## 14. 是否进入 constrained PPO multi-seed

只有当 augmented learned policy 满足第 13 节标准时，才建议下一步：

```text
006_13_multiseed_constrained_ppo_with_augmented_prior
```

如果 learned policy 仍不稳定，但 expert replay 很强，则建议：

```text
006_13_expert_policy_stability_and_rainfall_stress_test
```

不要直接做普通 PPO multi-seed。

---

## 15. 是否进入 rainfall-scaling budget scenario

只有在以下条件满足时，才建议 rainfall-scaling：

```text
1. expert / imitation policy 行为稳定；
2. 不依赖 300/450 cap；
3. HLA/SYA/LCA 至少通过；
4. yield loss 和 profit 可接受；
5. 已清楚说明该策略是 expert/imitation prior，不是自由 PPO 最优策略。
```

否则 rainfall-scaling 仍然过早。

---

## 16. 图表输出

至少生成：

```text
augmented_expert_schedule_pareto.png
augmented_dataset_action_distribution.png
original_vs_augmented_bc_actions.png
augmented_policy_yield_vs_input.png
augmented_policy_profit_comparison.png
site_level_augmented_policy_comparison.png
expert_type_yield_profit_tradeoff.png
```

保存到：

```text
Leave_One_experiments/expert_dataset_augmentation/figures/
```

---

## 17. 报告要求

报告必须说明：

1. 为什么不直接进入 PPO multi-seed；
2. 006_11 prior replay 修复结果；
3. expert dataset 如何扩充；
4. 新 dataset 的站点、年份、schedule 类型；
5. action 分布是否更丰富；
6. augmented BC 模型表现；
7. DSSAT/gym-DSSAT 实际评估；
8. 是否推荐 augmented learned prior；
9. 是否可以进入 constrained PPO multi-seed；
10. 是否可以进入 rainfall-scaling；
11. 如果不能，下一步建议。

---

## 18. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_expert_dataset_augmentation_report.md
```

生成 PPT：

```text
docs/2026-06-06_expert_dataset_augmentation_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/expert_dataset_augmentation/reports/
```

---

## 19. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/006_12_expert_dataset_augmentation_before_constrained_ppo.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/expert_dataset_augmentation/configs/
git add Leave_One_experiments/expert_dataset_augmentation/evaluation/
git add Leave_One_experiments/expert_dataset_augmentation/expert_policy/
git add Leave_One_experiments/expert_dataset_augmentation/imitation_dataset/
git add Leave_One_experiments/expert_dataset_augmentation/figures/
git add Leave_One_experiments/expert_dataset_augmentation/reports/
git add docs/2026-06-06_expert_dataset_augmentation_report.md
git add docs/2026-06-06_expert_dataset_augmentation_report.pptx
git commit -m "Augment expert dataset before constrained PPO"
```

注意：

1. 不要默认 commit 大模型；
2. 不要默认 commit tensorboard；
3. 不要默认 commit 大量 daily_outputs；
4. 不要强行 push。

---

## 20. 完成后请汇报

完成后请汇报：

1. 选入了多少 expert schedules；
2. augmented dataset 包含哪些站点和年份；
3. action 分布是否比 006_09 更丰富；
4. 训练了哪些 augmented BC models；
5. 哪些 policy 在 DSSAT/gym-DSSAT 中通过；
6. 推荐 policy 是 original BC、augmented BC，还是 expert replay；
7. 是否可以进入 constrained PPO multi-seed；
8. 是否可以进入 rainfall-scaling；
9. 如果不能，下一步应做什么。
