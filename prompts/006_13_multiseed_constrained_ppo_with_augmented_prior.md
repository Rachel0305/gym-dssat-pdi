# 006_13_multiseed_constrained_ppo_with_augmented_prior

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_11_debug_prior_replay_and_constrained_finetuning_consistency.md
prompts/006_12_expert_dataset_augmentation_before_constrained_ppo.md
docs/2026-06-06_prior_replay_debug_report.md
docs/2026-06-06_expert_dataset_augmentation_report.md
```

如果文件名或日期略有不同，请搜索关键词：

```text
expert_dataset_augmentation_report
BC_random_forest_regressor_augmented
recommended_augmented_prior_policy
augmented_imitation_policy_dssat_summary
```

现在执行新的子任务：只针对 `BC_random_forest_regressor_augmented` 这个推荐的 augmented learned prior，进行 constrained PPO multi-seed 稳定性验证。

本阶段不要训练普通 PPO，不要训练无 action safety PPO，不要进入 rainfall-scaling budget scenario，不要把所有方法都做 multi-seed。

---

## 1. 本阶段背景

上一阶段 `006_12_expert_dataset_augmentation_before_constrained_ppo` 已经完成。

关键结果：

1. expert dataset 已从 006_09 的 3 个 schedules 扩充到 45 个 schedules：
   - HLA: 15
   - LCA: 15
   - SYA: 15
2. 数据覆盖：
   - HLA: 2007, 2009, 2011
   - LCA: 2008, 2009, 2010, 2011
   - SYA: 2012, 2014, 2015
3. augmented dataset 明显比 006_09 更丰富：
   - 00609_original: 1300 rows, 3 schedules, nonzero_n_rows=14, nonzero_irrigation_rows=0
   - 00612_augmented: 19500 rows, 45 schedules, nonzero_n_rows=252, nonzero_irrigation_rows=87
4. augmented learned policies 中，通过 DSSAT/gym-DSSAT gate 的包括：
   - `BC_random_forest_regressor_augmented`
   - `BC_two_stage_classifier_regressor_augmented`
   - `BC_two_stage_classifier_regressor_original`
   - `best_expert_schedule_replay`
5. 最终推荐 policy 是：

```text
BC_random_forest_regressor_augmented
mean_yield = 6503.2767
mean_irrigation = 0.0
mean_n = 81.8291
mean_profit = 44.5755
mean_yield_loss_vs_ppo = 0.0542
max_irrigation = 0.0
max_n = 108.5183
mean_input_reduction_vs_ppo = 0.8909
passes_gate = True
recommendation = augmented_learned_policy
next_step = can_consider_00613_constrained_ppo_multiseed
```

因此，本阶段可以进入 constrained PPO multi-seed，但只能围绕 `BC_random_forest_regressor_augmented` 做稳定性验证。

---

## 2. 本阶段核心目标

本阶段目标是：

1. 使用 `BC_random_forest_regressor_augmented` 作为主 prior；
2. 只测试推荐的 constrained PPO 方法；
3. 做 multi-seed 稳定性验证；
4. 检查不同 seed 下是否仍避免 300/450 cap saturation；
5. 检查不同 seed 下水氮投入是否仍保持低投入；
6. 检查产量损失和 profit 是否稳定；
7. 比较 augmented BC prior、constrained PPO multi-seed、old PPO saturated baseline；
8. 判断是否可以进入下一阶段 rainfall-stress / irrigation-responsive expert search；
9. 如果 multi-seed 不稳定，则回到 expert dataset augmentation 或 stricter guardrail。

---

## 3. 本阶段不要做的事情

1. 不要训练普通 unrestricted PPO。
2. 不要训练无 action safety PPO。
3. 不要对所有 006_10 methods 做 multi-seed。
4. 不要使用 `BC_mlp_regressor_augmented`。
5. 不要把 `best_expert_schedule_replay` 当作 learned prior 做 PPO 初始化。
6. 不要进入 rainfall-scaling budget scenario。
7. 不要训练 FQA/YCA，除非 HLA/SYA/LCA multi-seed 完全稳定后报告中明确建议。
8. 不要覆盖 006_08 到 006_12 的结果。
9. 不要修改 `my_data/` 原始文件。
10. 不要默认 commit 大模型、tensorboard 或大量 daily outputs。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_expert_dataset_augmentation_report.md
Leave_One_experiments/expert_dataset_augmentation/evaluation/augmented_imitation_policy_dssat_summary.csv
Leave_One_experiments/expert_dataset_augmentation/evaluation/recommended_augmented_prior_policy.csv
Leave_One_experiments/expert_dataset_augmentation/models/
Leave_One_experiments/expert_dataset_augmentation/imitation_dataset/imitation_dataset_augmented.csv
Leave_One_experiments/expert_dataset_augmentation/expert_policy/HLA_augmented_expert_schedule_list.csv
Leave_One_experiments/expert_dataset_augmentation/expert_policy/SYA_augmented_expert_schedule_list.csv
Leave_One_experiments/expert_dataset_augmentation/expert_policy/LCA_augmented_expert_schedule_list.csv

docs/2026-06-06_prior_replay_debug_report.md
Leave_One_experiments/prior_replay_debug/evaluation/constrained_ppo_fixed_interface_summary.csv

src/train_imitation_policy.py
src/imitation_policy_models.py
src/evaluate_imitation_policy.py
src/replay_imitation_prior.py
src/episode_profit_reward.py
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/
```

建议目录结构：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/
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
docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.md
docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.pptx
```

---

## 6. 第一步：确认 prior 与 multi-seed 方案

请生成：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/multiseed_prior_and_method_plan.md
```

必须说明：

1. 为什么选择 `BC_random_forest_regressor_augmented`；
2. 为什么不选 `BC_mlp_regressor_augmented`；
3. 为什么 `BC_two_stage_classifier_regressor_augmented` 是备选而不是主 prior；
4. 为什么 rainfall-scaling 仍然过早；
5. 本阶段选择哪些 constrained PPO 方法；
6. 每个方法为什么适合 multi-seed；
7. 如果重新出现高投入退化，如何判定失败。

---

## 7. 第二步：选择 multi-seed 方法

本阶段不要测试太多方法。

优先测试：

```text
MS0_augmented_RF_prior_replay
MS1_residual_augmented_RF_prior_strict
MS2_guardrail_augmented_RF_100_200
```

说明：

### MS0_augmented_RF_prior_replay

不训练 PPO，直接 replay `BC_random_forest_regressor_augmented`。

作用：

```text
multi-seed 对照；理论上不同 seed 应完全一致或差异极小。
```

### MS1_residual_augmented_RF_prior_strict

PPO 只学习 residual action：

```text
final_action = action_augmented_RF_prior + residual_action
```

建议 residual 限制：

```text
irrigation_residual_range = [-10, 10] mm
n_residual_range = [-20, 20] kg/ha
```

season guardrail：

```text
season_irrigation_cap = 100 mm
season_n_cap = 200 kg/ha
```

### MS2_guardrail_augmented_RF_100_200

PPO 使用 imitation action penalty + strict guardrail：

```text
season_irrigation_cap = 100 mm
season_n_cap = 200 kg/ha
lambda_bc = 1.0
```

说明：

```text
这是防止 PPO 再次退化为高投入策略的保守版本。
```

不要在本阶段测试 300/450 cap，因为前面已经多次证明 PPO 容易回到高投入。

---

## 8. Multi-seed 设置

使用：

```text
seeds = [0, 1, 2, 3, 4]
timesteps = 5000
```

如果计算量太大，先运行：

```text
seeds = [0, 1, 2]
```

并在报告中说明未运行完整 5 seeds 的原因。

---

## 9. 站点和年份

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

暂时不要做 FQA/YCA。

原因：

```text
FQA/YCA only have two-year data, and augmented expert prior is currently built and validated mainly on HLA/SYA/LCA.
```

---

## 10. Action safety 与 reward 设置

统一使用：

```text
action_safety_enabled = true
daily_irrigation_max = 40 mm
daily_n_max = 80 kg/ha
```

对于 MS1/MS2，使用 strict season guardrail：

```text
season_irrigation_cap = 100 mm
season_n_cap = 200 kg/ha
```

reward 使用：

```text
episode-level profit reward
+ imitation penalty 或 residual constraint
```

profit score 与前面保持一致：

```text
profit_score = 0.01 * final_grnwt - 0.5 * total_irrigation - 0.25 * total_n
```

如果 terminal profit reward 与 PPO 训练接口不稳定，可以使用 existing reward + imitation penalty，但报告必须说明。

---

## 11. 每个 run 前 smoke check

每个 seed/method/station run 前必须做：

```text
null_zero
prior_replay
fixed_low_input
```

保存：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/pretrain_smoke_check_summary.csv
```

如果 smoke check 失败，不训练该 run。

---

## 12. 每个 policy 评估

每个 policy 训练后，在 train_year 和 eval_years 上评估。

summary 保存到：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/multiseed_constrained_ppo_summary.csv
```

字段至少包括：

```text
station
method
seed
prior_policy
train_year
eval_year
run_status
episode_completed
final_grnwt
total_irrigation
total_n_fertilizer
profit_score
mean_swfac
mean_nstres
yield_loss_vs_augmented_rf_prior
yield_loss_vs_expert_best
yield_loss_vs_ppo_baseline
input_reduction_vs_ppo_baseline
bc_action_deviation_mean
bc_action_deviation_max
irrigation_saturation_ratio_100_200
n_saturation_ratio_100_200
irrigation_saturation_ratio_300_450
n_saturation_ratio_300_450
daily_csv_path
notes
```

---

## 13. Multi-seed 汇总指标

按 site + method 汇总：

```text
mean_yield
std_yield
cv_yield
mean_profit
std_profit
mean_irrigation
std_irrigation
mean_n
std_n
mean_yield_loss_vs_ppo
mean_input_reduction_vs_ppo
failure_rate
cap_regression_rate
```

输出：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/multiseed_method_comparison.csv
```

---

## 14. 推荐 method 通过标准

推荐 method 必须满足：

```text
run_status ok for all seeds and eval years
mean_irrigation <= 100 mm
mean_n <= 200 kg/ha
mean_yield_loss_vs_ppo_baseline <= 15%
mean_profit >= 90% of augmented RF prior profit
cap_regression_rate = 0
yield CV acceptable
behavior interpretable
```

如果 MS1/MS2 相比 MS0 产量没有提升或 profit 下降明显，则推荐保留 augmented RF prior，不推荐 PPO fine-tuning。

如果任何 method 重新接近 300/450，标记：

```text
regression_to_high_input_policy
```

---

## 15. 策略比较

必须比较：

```text
best_expert_schedule_replay
BC_random_forest_regressor_augmented
BC_two_stage_classifier_regressor_original
MS0_augmented_RF_prior_replay
MS1_residual_augmented_RF_prior_strict
MS2_guardrail_augmented_RF_100_200
old_ppo_cap_saturated_baseline
```

输出：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/policy_comparison_multiseed_augmented_prior.csv
```

---

## 16. 图表输出

至少生成：

```text
multiseed_yield_boxplot_by_method.png
multiseed_profit_boxplot_by_method.png
multiseed_input_boxplot_by_method.png
multiseed_yield_vs_input.png
cap_regression_rate_by_method.png
site_level_multiseed_policy_comparison.png
prior_vs_finetuned_daily_actions.png
```

保存到：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/figures/
```

---

## 17. 是否进入 rainfall-scaling

本阶段结束后，只有在以下条件全部满足时，才建议进入 rainfall-scaling：

```text
1. 推荐 method 或 augmented RF prior 在 HLA/SYA/LCA 稳定；
2. 不依赖 300/450 cap；
3. mean irrigation <= 100 或明确解释为什么为 0；
4. mean N <= 200；
5. yield loss <= 15%；
6. profit 稳定；
7. multi-seed 无明显退化。
```

但注意：

```text
006_12 报告指出 irrigation events 仍然稀疏，因此 rainfall-scaling 前更合理的下一步可能是 irrigation-responsive expert search。
```

所以如果 irrigation 仍然全为 0，不要直接做 rainfall-scaling，而应先设计：

```text
006_14_irrigation_responsive_expert_search
```

---

## 18. 报告要求

报告必须说明：

1. 为什么 `BC_random_forest_regressor_augmented` 是主 prior；
2. multi-seed 方法设计；
3. seeds 和站点；
4. 每个方法是否稳定；
5. 是否重新发生高投入退化；
6. 是否比 augmented RF prior 更好；
7. 推荐保留 prior 还是推荐 PPO fine-tuned method；
8. 是否可以进入下一阶段；
9. 是否可以进入 rainfall-scaling；
10. 如果不能，是否应做 irrigation-responsive expert search。

---

## 19. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.md
```

生成 PPT：

```text
docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/reports/
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
git add prompts/006_13_multiseed_constrained_ppo_with_augmented_prior.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/configs/
git add Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/evaluation/
git add Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/figures/
git add Leave_One_experiments/constrained_ppo_multiseed_augmented_prior/reports/
git add docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.md
git add docs/2026-06-06_constrained_ppo_multiseed_augmented_prior_report.pptx
git commit -m "Run multiseed constrained PPO with augmented prior"
```

注意：

1. 不要默认 commit 大模型；
2. 不要默认 commit tensorboard；
3. 不要默认 commit 大量 daily_outputs；
4. 不要强行 push。

---

## 21. 完成后请汇报

完成后请汇报：

1. 跑了哪些 seeds；
2. 跑了哪些 methods；
3. HLA/SYA/LCA 是否全部通过；
4. 是否有 method 重新退化到 300/450；
5. 是否有 method 比 augmented RF prior 更好；
6. 推荐保留 prior 还是使用 PPO fine-tuned method；
7. 是否可以进入下一阶段；
8. 下一阶段是 rainfall-scaling 还是 irrigation-responsive expert search。
