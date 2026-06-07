# 006_10_constrained_ppo_finetuning_from_imitation_prior

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_08_offline_schedule_search_and_bayesian_optimization_prior.md
prompts/006_09_train_imitation_learning_prior_from_offline_schedules.md
docs/2026-06-06_offline_schedule_search_report.md
docs/2026-06-06_imitation_learning_prior_report.md
```

现在执行新的子任务：基于 imitation learning prior 进行 constrained PPO fine-tuning。目标不是回到无约束 PPO，而是在 expert prior 附近小幅优化，避免再次退化为 300 mm / 450 kg ha-1 N cap saturation。

本阶段不要做普通 PPO multi-seed，不要进入 rainfall-scaling budget scenario，不要训练无 action safety 的 PPO，不要覆盖 006_03 到 006_09 的结果。

---

## 1. 本阶段背景

上一阶段 `006_09_train_imitation_learning_prior_from_offline_schedules` 已完成。主要结果：

```text
BC_constant_schedule_baseline:
  mean_yield = 8225.2271
  mean_irrigation = 0.0
  mean_n = 150.0
  mean_yield_loss_vs_ppo = -0.1962

BC_random_forest_regressor:
  mean_yield = 6115.2526
  mean_irrigation = 0.0
  mean_n = 85.8948
  mean_yield_loss_vs_ppo = 0.1106

BC_two_stage_classifier_regressor:
  mean_yield = 8091.8537
  mean_irrigation = 0.0
  mean_n = 153.575
  mean_yield_loss_vs_ppo = -0.1768
```

通过 constrained PPO fine-tuning gate 的 policies：

```text
BC_constant_schedule_baseline
BC_random_forest_regressor
BC_two_stage_classifier_regressor
```

`BC_constant_schedule_baseline` 是 expert replay，不是 learned general policy；`BC_two_stage_classifier_regressor` 是更合适的 learned prior；`BC_random_forest_regressor` 作为备选 prior；`BC_mlp_regressor` 产量过低，不推荐。

---

## 2. 本阶段目标

1. 基于 imitation prior 初始化或约束 PPO。
2. 避免 PPO fine-tuning 后重新打满 300/450 cap。
3. 检查 PPO 是否能在 expert prior 基础上小幅提高 yield 或 profit。
4. 保持水氮投入在合理范围内。
5. 比较 expert schedule、BC prior、constrained PPO fine-tuned policy、旧 PPO cap-saturated baseline。
6. 只先做 HLA/SYA/LCA，不做 FQA/YCA。
7. 只先做 seed=0，不做 multi-seed。
8. 如果 seed=0 稳定通过，再考虑后续多 seed。

---

## 3. 禁止事项

1. 不要训练无 action safety PPO。
2. 不要做普通 PPO multi-seed。
3. 不要进入 rainfall-scaling budget scenario。
4. 不要训练 FQA/YCA。
5. 不要覆盖 006_03 到 006_09 的结果。
6. 不要修改 `my_data/` 原始文件。
7. 不要回到 daily unrestricted PPO。
8. 不要让 PPO 使用 300/450 作为默认目标。
9. 不要只看 reward，必须同时检查 yield、water、N、profit、saturation ratio。
10. 不要默认 commit 大模型或大量 daily outputs。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_imitation_learning_prior_report.md
Leave_One_experiments/imitation_learning_prior/evaluation/imitation_policy_dssat_evaluation_summary.csv
Leave_One_experiments/imitation_learning_prior/evaluation/policy_comparison_expert_bc_ppo.csv
Leave_One_experiments/imitation_learning_prior/evaluation/imitation_supervised_metrics.csv
Leave_One_experiments/imitation_learning_prior/models/
Leave_One_experiments/imitation_learning_prior/datasets/imitation_dataset_clean.csv
Leave_One_experiments/offline_schedule_search/expert_policy/HLA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/SYA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/LCA_expert_schedule_ranking.csv
src/train_imitation_policy.py
src/imitation_policy_models.py
src/evaluate_imitation_policy.py
src/episode_profit_reward.py
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
sb3_wrapper.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/constrained_ppo_finetuning/
```

建议目录结构：

```text
Leave_One_experiments/constrained_ppo_finetuning/
  configs/
  prior_policies/
  models/
  logs/
  tensorboard/
  daily_outputs/
  evaluation/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-06_constrained_ppo_finetuning_report.md
docs/2026-06-06_constrained_ppo_finetuning_report.pptx
```

---

## 6. 第一步：确认 fine-tuning prior

请生成：

```text
Leave_One_experiments/constrained_ppo_finetuning/evaluation/prior_policy_selection.md
```

必须说明：

1. 为什么不选 `BC_constant_schedule_baseline` 作为主要 learned prior；
2. 为什么 `BC_two_stage_classifier_regressor` 是主推荐；
3. 为什么 `BC_random_forest_regressor` 是备选；
4. 为什么 `BC_mlp_regressor` 不推荐；
5. imitation policy 是否避免 300/450 saturation；
6. prior 的平均水氮投入和产量；
7. prior 的局限：专家动作稀疏，irrigation 全为 0，主要学习的是 nitrogen event schedule。

---

## 7. 第二步：设计 constrained PPO fine-tuning 方案

至少设计 3 种 fine-tuning 方式。

### FT0: prior replay baseline

不训练 PPO，直接复现 prior policy。

```text
FT0_BC_two_stage_replay
```

### FT1: PPO with imitation action penalty

PPO 正常输出动作，但 reward 中加入 action deviation penalty，使其不要偏离 BC prior 太远。

```text
reward_total = environment_reward + terminal_profit_reward - lambda_bc * ||action_ppo - action_bc||^2
```

建议测试：

```text
lambda_bc = 0.1
lambda_bc = 0.5
lambda_bc = 1.0
```

### FT2: PPO residual action around prior

PPO 不直接输出完整动作，而是输出 residual：

```text
final_action = action_bc + residual_action
irrigation_residual_range = [-20, 20] mm
n_residual_range = [-30, 30] kg/ha
```

### FT3: PPO with budget guardrail

在 prior 基础上训练 PPO，但增加比 300/450 更严格的 guardrail：

```text
season_irrigation_cap = 100 mm
season_n_cap = 200 kg/ha
```

---

## 8. 推荐优先测试顺序

先测试 HLA/SYA/LCA 三站点 seed=0。

优先顺序：

```text
FT0_BC_two_stage_replay
FT1_bc_penalty_lambda_0.5
FT2_residual_bc_prior
FT3_budget_guardrail_100_200
```

如果计算量允许，再补：

```text
FT1_bc_penalty_lambda_0.1
FT1_bc_penalty_lambda_1.0
```

不要一开始做多 seed。

---

## 9. 训练站点和年份

本阶段只做 HLA/SYA/LCA。

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

FQA/YCA 暂时不做，因为它们只有 two-year data，且 imitation expert prior 尚未覆盖。

---

## 10. 训练设置

```text
seed = 0
total_timesteps = 5000
action_safety_enabled = true
```

对于 FT1/FT2：

```text
season_irrigation_cap = 300 mm
season_n_cap = 450 kg/ha
```

但必须额外检查是否退化到高投入。

对于 FT3：

```text
season_irrigation_cap = 100 mm
season_n_cap = 200 kg/ha
```

daily max 建议：

```text
daily_irrigation_max = 40 mm
daily_n_max = 80 kg/ha
```

reward 使用：

```text
episode-level profit reward + imitation penalty 或 residual constraint
```

如果 terminal profit reward 与 PPO 训练接口不稳定，请使用 BC imitation penalty + existing reward，并在报告中说明。

---

## 11. 每个 fine-tuning run 前 smoke check

每个 run 前必须做：

```text
null_zero
prior_replay
fixed_low_input
```

保存到：

```text
Leave_One_experiments/constrained_ppo_finetuning/evaluation/pretrain_smoke_check_summary.csv
```

如果 smoke check 失败，不训练该 run。

---

## 12. 每个 fine-tuned policy 评估

每个 policy 训练后，在 train_year 和 eval_years 上评估。

保存 daily CSV 到：

```text
Leave_One_experiments/constrained_ppo_finetuning/daily_outputs/
```

summary 保存到：

```text
Leave_One_experiments/constrained_ppo_finetuning/evaluation/constrained_ppo_evaluation_summary.csv
```

字段至少包括：

```text
station
policy_name
finetune_method
prior_policy
train_year
eval_year
seed
run_status
episode_completed
final_grnwt
total_irrigation
total_n_fertilizer
mean_swfac
mean_nstres
profit_score
yield_loss_vs_prior
yield_loss_vs_expert
yield_loss_vs_ppo_baseline
input_reduction_vs_ppo_baseline
bc_action_deviation_mean
bc_action_deviation_max
irrigation_saturation_ratio_300_450
n_saturation_ratio_300_450
irrigation_saturation_ratio_guardrail
n_saturation_ratio_guardrail
daily_csv_path
notes
```

---

## 13. 策略比较

必须比较：

```text
expert_schedule
BC_two_stage_classifier_regressor
BC_random_forest_regressor
FT0_BC_two_stage_replay
FT1_bc_penalty
FT2_residual_bc_prior
FT3_budget_guardrail
old_ppo_cap_saturated_baseline
```

输出：

```text
Leave_One_experiments/constrained_ppo_finetuning/evaluation/policy_comparison_expert_bc_finetune_ppo.csv
```

重点判断：

1. fine-tuned PPO 是否提高 yield 或 profit；
2. 是否保持 low input；
3. 是否重新接近 300/450；
4. 是否比 BC prior 更稳定；
5. 是否比 old PPO baseline 更有经济意义。

---

## 14. 通过标准

本阶段推荐 policy 必须满足：

```text
run_status ok for all eval years
mean_irrigation <= 100 mm
mean_n <= 200 kg/ha
mean_yield_loss_vs_ppo_baseline <= 15%
mean_profit_score >= BC_two_stage profit_score 或接近
not saturated at 300/450
```

如果 fine-tuned PPO 比 BC prior 投入显著增加但产量提升很小，应判定为不推荐。

如果 fine-tuned PPO 又回到 300/450，应判定为 regression_to_saturated_policy。

---

## 15. 图表输出

至少生成：

```text
constrained_ppo_yield_vs_input.png
constrained_ppo_profit_comparison.png
constrained_ppo_inputs_by_policy.png
bc_deviation_distribution.png
expert_bc_finetune_daily_actions.png
expert_bc_finetune_cumulative_inputs.png
site_level_policy_comparison.png
```

保存到：

```text
Leave_One_experiments/constrained_ppo_finetuning/figures/
```

---

## 16. 是否进入多 seed

如果 seed=0 下至少一个 fine-tuning method 通过标准，报告建议下一步：

```text
006_11_multiseed_constrained_ppo_finetuning
```

多 seed 只对推荐 method 做，不要对所有 methods 做。

如果没有 method 通过，不要做多 seed。应回到 expert dataset augmentation 或 offline schedule search。

---

## 17. 是否进入 rainfall-scaling budget scenario

只有在以下条件满足时，才建议进入 rainfall-scaling：

```text
1. recommended constrained PPO policy 不再打满 300/450；
2. HLA/SYA/LCA 都通过；
3. yield loss <= 15%；
4. policy behavior 可解释；
5. multi-seed 稳定性已准备或已完成。
```

否则 rainfall-scaling 仍然过早。

---

## 18. 报告要求

报告必须说明：

1. 为什么进入 constrained PPO，而不是普通 PPO；
2. prior policy 如何选择；
3. fine-tuning methods；
4. HLA/SYA/LCA 结果；
5. 是否重新发生 cap saturation；
6. 与 expert schedule、BC prior、old PPO baseline 的对比；
7. 推荐 method；
8. 是否可以进入多 seed；
9. 是否可以进入 rainfall-scaling；
10. 如果失败，下一步应做什么。

---

## 19. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_constrained_ppo_finetuning_report.md
```

生成 PPT：

```text
docs/2026-06-06_constrained_ppo_finetuning_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/constrained_ppo_finetuning/reports/
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
git add prompts/006_10_constrained_ppo_finetuning_from_imitation_prior.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/constrained_ppo_finetuning/configs/
git add Leave_One_experiments/constrained_ppo_finetuning/evaluation/
git add Leave_One_experiments/constrained_ppo_finetuning/figures/
git add Leave_One_experiments/constrained_ppo_finetuning/reports/
git add docs/2026-06-06_constrained_ppo_finetuning_report.md
git add docs/2026-06-06_constrained_ppo_finetuning_report.pptx
git commit -m "Fine tune constrained PPO from imitation prior"
```

注意：不要默认 commit 大模型；不要默认 commit tensorboard；不要默认 commit 大量 daily_outputs；不要强行 push。

---

## 21. 完成后请汇报

完成后请汇报：

1. 选择了哪个 prior；
2. 测试了哪些 fine-tuning methods；
3. 哪些 method 避免了 300/450 饱和；
4. 哪些 method 满足 mean irrigation <=100 和 mean N <=200；
5. 是否比 BC prior 提高 yield/profit；
6. 推荐 method 是哪个；
7. 是否可以进入 constrained PPO multi-seed；
8. 是否可以进入 rainfall-scaling budget scenario；
9. 如果失败，是否需要 expert dataset augmentation。