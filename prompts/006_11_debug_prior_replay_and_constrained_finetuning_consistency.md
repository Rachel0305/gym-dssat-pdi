# 006_11_debug_prior_replay_and_constrained_finetuning_consistency

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_09_train_imitation_learning_prior_from_offline_schedules.md
prompts/006_10_constrained_ppo_finetuning_from_imitation_prior.md
docs/2026-06-06_imitation_learning_prior_report.md
docs/2026-06-06_constrained_ppo_finetuning_report.md
```

如果文件名或日期略有不同，请搜索关键词：

```text
imitation_learning_prior_report
constrained_ppo_finetuning_report
BC_two_stage_classifier_regressor
FT0_BC_two_stage_replay
```

现在执行新的子任务：先 debug `006_09` imitation policy evaluation 与 `006_10` FT0 prior replay 之间的不一致，确认 BC prior 能被稳定复现后，再重新进行 constrained PPO fine-tuning 小批量测试。

本阶段不要做 multi-seed，不要进入 rainfall-scaling budget scenario，不要训练无 action safety 的 PPO，不要训练 FQA/YCA。

---

## 1. 本阶段背景

`006_09` 的 imitation learning prior 报告显示：

```text
BC_two_stage_classifier_regressor:
  eval_count = 10
  mean_yield = 8091.8537
  mean_irrigation = 0.0
  mean_n = 153.575
  mean_yield_loss_vs_ppo = -0.1768
```

该模型通过了 constrained PPO fine-tuning gate。

但是 `006_10` 的 constrained PPO fine-tuning 报告中：

```text
FT0_BC_two_stage_replay:
  eval_count = 10
  mean_yield = 4447.8684
  mean_irrigation = 0.0
  mean_n = 56.0
  mean_yield_loss_vs_ppo = 0.3531
```

这与 `006_09` 的 BC_two_stage evaluation 明显不一致。

这说明 `006_10` 中的 FT0 prior replay 很可能没有正确复现 `006_09` 的 BC_two_stage policy。可能原因包括：

1. 使用了不同的模型文件；
2. 使用了不同的 action scaling；
3. 使用了不同的 wrapper 顺序；
4. 使用了不同的 action safety cap；
5. 使用了不同的 eval years；
6. prior replay 时没有正确执行非零施氮事件；
7. daily action 被 residual/guardrail/wrapper 意外覆盖；
8. BC two-stage classifier/regressor 的 event classifier 阈值或后处理不同；
9. `expert_action_n` 与模型输出单位不一致；
10. `FT0` 实现不是纯 replay，而被 PPO/guardrail 或其他逻辑改写。

因此，本阶段的核心不是继续调 PPO，而是先确认：

```text
006_10 的 FT0_BC_two_stage_replay 是否能逐日复现 006_09 的 BC_two_stage_classifier_regressor 输出。
```

---

## 2. 本阶段目标

1. 对比 `006_09` 和 `006_10` 的 BC_two_stage evaluation 结果；
2. 找出 FT0 replay 与 BC_two_stage policy 不一致的原因；
3. 修复 FT0 prior replay；
4. 重新评估 FT0；
5. 只有当 FT0 与 `006_09` 结果一致后，才重新运行小批量 constrained PPO fine-tuning；
6. 不做 multi-seed；
7. 不做 rainfall-scaling；
8. 不训练 FQA/YCA。

---

## 3. 本阶段不要做的事情

1. 不要做普通 PPO multi-seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练无 action safety PPO。
4. 不要训练 FQA/YCA。
5. 不要覆盖 006_09 或 006_10 结果。
6. 不要直接把 006_10 的失败解释为 PPO 失败，必须先排查 FT0 replay 不一致。
7. 不要只看 summary，必须检查 daily action。
8. 不要默认 commit 大模型和大量 daily outputs。
9. 不要修改 `my_data/` 原始文件。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_imitation_learning_prior_report.md
docs/2026-06-06_constrained_ppo_finetuning_report.md

Leave_One_experiments/imitation_learning_prior/evaluation/imitation_policy_dssat_evaluation_summary.csv
Leave_One_experiments/imitation_learning_prior/evaluation/policy_comparison_expert_bc_ppo.csv
Leave_One_experiments/imitation_learning_prior/evaluation/imitation_supervised_metrics.csv
Leave_One_experiments/imitation_learning_prior/models/
Leave_One_experiments/imitation_learning_prior/datasets/imitation_dataset_clean.csv

Leave_One_experiments/constrained_ppo_finetuning/evaluation/constrained_ppo_evaluation_summary.csv
Leave_One_experiments/constrained_ppo_finetuning/evaluation/policy_comparison_expert_bc_finetune_ppo.csv
Leave_One_experiments/constrained_ppo_finetuning/daily_outputs/

src/imitation_policy_models.py
src/evaluate_imitation_policy.py
src/train_imitation_policy.py
src/episode_profit_reward.py
src/ppo_action_safety.py
src/ppo_evaluate.py
src/ppo_train.py
src/ppo_safe_rendering.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/prior_replay_debug/
```

建议目录结构：

```text
Leave_One_experiments/prior_replay_debug/
  configs/
  daily_outputs/
  evaluation/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-06_prior_replay_debug_report.md
docs/2026-06-06_prior_replay_debug_report.pptx
```

---

## 6. 第一步：生成 prior replay mismatch review

请生成：

```text
Leave_One_experiments/prior_replay_debug/evaluation/prior_replay_mismatch_review.md
```

必须逐项对比：

```text
006_09 BC_two_stage_classifier_regressor
006_10 FT0_BC_two_stage_replay
```

对比内容至少包括：

1. eval years 是否一致；
2. station list 是否一致；
3. 模型文件路径是否一致；
4. 数据标准化器是否一致；
5. feature columns 是否一致；
6. action post-processing 是否一致；
7. action unit 是否一致；
8. wrapper 顺序是否一致；
9. action safety cap 是否一致；
10. daily max 是否一致；
11. 是否使用 same expert schedule baseline；
12. 是否把 predicted N action 裁剪或归零；
13. 是否把 event classifier threshold 设置不同；
14. 是否有 residual/guardrail 意外影响 FT0。

---

## 7. 第二步：逐日 action 对齐检查

对 HLA/SYA/LCA 的全部 eval years，输出 daily action 对齐表：

```text
Leave_One_experiments/prior_replay_debug/evaluation/bc_two_stage_vs_ft0_daily_action_alignment.csv
```

字段至少包括：

```text
station
eval_year
date
dap
bc_two_stage_irrigation_00609
bc_two_stage_n_00609
ft0_irrigation_00610
ft0_n_00610
expert_schedule_irrigation
expert_schedule_n
irrigation_diff
n_diff
is_event_day
notes
```

重点检查：

1. 非事件日是否都为 0；
2. 事件日是否出现正确施氮量；
3. HLA/SYA/LCA 的 event DAP 是否一致；
4. FT0 是否漏掉某些施氮事件；
5. FT0 的 total_n 为什么只有 56 而不是约 153；
6. 是否是平均到 daily 的单位问题；
7. 是否是 normalized action 没有反归一化。

---

## 8. 第三步：重新实现纯 replay prior evaluator

请新增或更新：

```text
src/replay_imitation_prior.py
```

要求实现一个“纯 replay / 纯 inference”的 evaluator：

1. 加载 `BC_two_stage_classifier_regressor`；
2. 使用与 `006_09` 完全一致的 feature columns 和 scaler；
3. 对每天状态预测 action；
4. 只做必要的 action clipping；
5. 不接 PPO；
6. 不接 residual action；
7. 不接 imitation penalty；
8. action safety 只作为最后安全阀；
9. 输出 daily CSV；
10. 输出 summary。

输出目录：

```text
Leave_One_experiments/prior_replay_debug/daily_outputs/
Leave_One_experiments/prior_replay_debug/evaluation/replayed_bc_two_stage_evaluation_summary.csv
```

summary 字段至少包括：

```text
station
eval_year
run_status
episode_completed
final_grnwt
total_irrigation
total_n_fertilizer
mean_swfac
mean_nstres
profit_score
yield_loss_vs_00609_bc_two_stage
n_diff_vs_00609_bc_two_stage
irrigation_diff_vs_00609_bc_two_stage
daily_csv_path
notes
```

---

## 9. 第四步：通过标准

纯 replay prior 必须满足：

```text
mean_irrigation = 0.0 或与 006_09 差异很小
mean_n 与 006_09 BC_two_stage mean_n 的相对差异 <= 5%
mean_yield 与 006_09 BC_two_stage mean_yield 的相对差异 <= 5%
run_status ok for all eval years
```

如果不满足，不要继续 PPO fine-tuning。

如果不满足，必须生成：

```text
Leave_One_experiments/prior_replay_debug/evaluation/prior_replay_blocker_report.md
```

说明：

1. 不一致来自哪里；
2. 需要修哪个文件；
3. 是否是 action scaling；
4. 是否是 wrapper 顺序；
5. 是否是模型加载问题；
6. 是否需要回到 006_09 重建 imitation dataset。

---

## 10. 第五步：修复 006_10 constrained PPO 接口

只有当纯 replay prior 通过后，才修复 constrained PPO fine-tuning 接口。

请检查并修复：

```text
src/ppo_train.py
src/ppo_evaluate.py
src/constrained_ppo_finetune.py
src/imitation_policy_models.py
src/evaluate_imitation_policy.py
```

具体要求：

1. FT0 必须调用同一个 replay evaluator；
2. FT1/FT2/FT3 必须使用同一套 prior action；
3. BC prior action 必须以真实单位传给 penalty/residual；
4. PPO action 与 BC action 比较前必须处于同一尺度；
5. residual action 的最终执行量必须记录；
6. guardrail 不得意外覆盖 FT0。

---

## 11. 第六步：重新运行最小 constrained PPO fine-tuning

在 FT0 replay 修复通过后，只重新运行最小集合：

```text
FT0_BC_two_stage_replay_fixed
FT2_residual_bc_prior_fixed
FT3_budget_guardrail_100_200_fixed
```

不要重跑所有 FT1 lambda。

站点：

```text
HLA, SYA, LCA
```

年份：

```text
HLA:
  train_year = 2011
  eval_years = 2011, 2007, 2009

SYA:
  train_year = 2012
  eval_years = 2012, 2014, 2015

LCA:
  train_year = 2010
  eval_years = 2010, 2011, 2008, 2009
```

训练设置：

```text
seed = 0
timesteps = 5000
```

保存 summary 到：

```text
Leave_One_experiments/prior_replay_debug/evaluation/constrained_ppo_fixed_interface_summary.csv
```

---

## 12. 重新运行后的通过标准

推荐 method 必须满足：

```text
run_status ok for all eval years
mean_irrigation <= 100 mm
mean_n <= 200 kg/ha
mean_yield_loss_vs_ppo_baseline <= 15%
not saturated at 300/450
profit_score not much worse than BC_two_stage prior
```

如果 FT2/FT3 不如 FT0 replay，不要推荐 fine-tuned PPO；应推荐保留 BC prior 或做 expert dataset augmentation。

---

## 13. 图表输出

至少生成：

```text
prior_replay_00609_vs_00610_total_inputs.png
prior_replay_daily_action_alignment.png
prior_replay_yield_comparison.png
fixed_constrained_ppo_yield_vs_input.png
fixed_constrained_ppo_profit_comparison.png
fixed_constrained_ppo_bc_deviation.png
```

保存到：

```text
Leave_One_experiments/prior_replay_debug/figures/
```

---

## 14. 报告要求

报告必须说明：

1. 为什么 006_10 不能直接进入 multi-seed；
2. 006_09 与 006_10 的不一致是什么；
3. mismatch 原因；
4. 修复了哪些文件；
5. 纯 replay prior 是否复现 006_09；
6. 修复后 FT0/FT2/FT3 结果；
7. 是否有推荐 fine-tuning method；
8. 是否可以进入 constrained PPO multi-seed；
9. 如果不能，下一步是 expert dataset augmentation 还是保留 BC prior。

---

## 15. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_prior_replay_debug_report.md
```

生成 PPT：

```text
docs/2026-06-06_prior_replay_debug_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/prior_replay_debug/reports/
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
git add prompts/006_11_debug_prior_replay_and_constrained_finetuning_consistency.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/prior_replay_debug/configs/
git add Leave_One_experiments/prior_replay_debug/evaluation/
git add Leave_One_experiments/prior_replay_debug/figures/
git add Leave_One_experiments/prior_replay_debug/reports/
git add docs/2026-06-06_prior_replay_debug_report.md
git add docs/2026-06-06_prior_replay_debug_report.pptx
git commit -m "Debug prior replay consistency for constrained PPO"
```

注意：

1. 不要默认 commit 大模型；
2. 不要默认 commit tensorboard；
3. 不要默认 commit 大量 daily_outputs；
4. 不要强行 push。

---

## 17. 完成后请汇报

完成后请汇报：

1. 006_09 BC_two_stage 与 006_10 FT0 的差异原因；
2. 是否修复 prior replay；
3. 纯 replay 是否复现 006_09；
4. 修复后 FT2/FT3 是否通过；
5. 推荐 method 是什么；
6. 是否可以进入 constrained PPO multi-seed；
7. 如果不能，是否需要 expert dataset augmentation。
