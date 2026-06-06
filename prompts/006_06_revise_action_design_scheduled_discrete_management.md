# 006_06_revise_action_design_scheduled_discrete_management

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_03_reward_cost_revision_and_debug.md
prompts/006_04_tune_reward_cost_coefficients_until_unsaturated.md
prompts/006_05_redesign_episode_level_profit_reward.md
docs/2026-06-06_reward_cost_revision_and_debug_report.md
docs/2026-06-06_reward_cost_coefficient_tuning_report.md
docs/2026-06-06_episode_profit_reward_debug_report.md
```

现在执行新的子任务：修改 PPO 的 action design，从“每日连续水氮动作”转向“低频/阶段性/预算式管理动作”，先做 HLA 2011 pilot。

本阶段不要做多 seed，不要进入 rainfall-scaling budget scenario，不要训练五站点，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

前面三轮 reward 调试已经证明：

1. 006_03：A/B/C reward candidates 都打满 300 mm irrigation / 450 kg ha-1 N。
2. 006_04：D/E/F/G 更强成本 candidates 仍然全部打满 300/450。
3. 006_05：terminal episode-level profit reward 已经可以在 wrapper 层实现，但 P0-P6 仍然全部打满 300/450。
4. terminal reward feasibility 已确认：`EpisodeProfitRewardWrapper` 可以在 done 后读取 final `grnwt`，并累计 safe `amir` / `anfer`。
5. 但所有 P candidates 的 mean_irrigation = 300、mean_n = 450、saturation ratio = 1.0。
6. 报告结论指出：如果所有 P candidates 仍然 hit cap，剩余问题很可能不是 reward coefficient，而是 action design，例如 scheduled discrete management events、lower-frequency decisions 或 explicit season budget actions。

因此，本阶段不再继续调 reward 系数，而是重构 action design。

---

## 2. 当前问题的解释

当前 PPO 的 action design 是每日连续动作：

```text
每天都可以输出 amir
每天都可以输出 anfer
```

即使有 action safety，PPO 仍然每天都有机会尝试施加水氮，最终通过 safety cap 被拦截到 season cap。这会诱导 PPO 学到：

```text
只要每天尽量给，最终一定会用满 cap
```

这不是农学管理上合理的决策频率。真实农田管理更接近少数几个关键生育期进行灌溉/施肥决策，或者先决定季节总预算，再决定在几个窗口如何分配。

---

## 3. 本阶段目标

1. 诊断当前 daily continuous action design 为什么导致 cap saturation。
2. 设计至少两种新的 action design。
3. 优先实现低侵入版本，不修改 DSSAT 源码。
4. 只在 HLA 2011 上做 pilot。
5. 使用已经可实现的 episode-level profit reward 或当前最可解释 reward。
6. 检查新 action design 是否能让 PPO 不再打满 300/450 cap。
7. 检查产量损失是否可接受。
8. 如果 HLA 成功，再扩展 SYA/LCA 小测试。
9. 如果 action design 仍失败，再报告需要改变 action space 或转为 scripted/budget allocation policy。

---

## 4. 本阶段不要做的事情

1. 不要做多 seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练 FQA/YCA。
4. 不要训练五站点。
5. 不要训练无 action safety 的 PPO。
6. 不要修改 `my_data/` 原始文件。
7. 不要覆盖 site-packages 中的原始 reward 文件。
8. 不要覆盖 006_03、006_04、006_05 结果。
9. 不要继续只调 reward 系数。
10. 不要大改 gym-DSSAT 或 DSSAT Fortran 源码。
11. 不要把本阶段结果写成最终论文结论。

---

## 5. 输入文件

优先读取：

```text
docs/2026-06-06_episode_profit_reward_debug_report.md
Leave_One_experiments/episode_profit_reward_debug/evaluation/episode_profit_reward_evaluation_summary.csv
Leave_One_experiments/episode_profit_reward_debug/evaluation/episode_profit_reward_candidate_comparison.csv
src/episode_profit_reward.py
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
sb3_wrapper.py
```

同时读取上一阶段 action safety 和 cap 相关配置：

```text
experiments/ppo_observed_years/config_season_cap_sensitivity.yaml
experiments/ppo_observed_years/config_ppo_action_safe_site_training.yaml
```

如果路径不同，请搜索文件名，不要猜。

---

## 6. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/action_design_debug/
```

建议目录结构：

```text
Leave_One_experiments/action_design_debug/
  configs/
  wrappers/
  smoke_checks/
  rendered_inputs/
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
docs/2026-06-06_action_design_debug_report.md
docs/2026-06-06_action_design_debug_report.pptx
```

---

## 7. 第一步：诊断 daily action saturation

请生成：

```text
Leave_One_experiments/action_design_debug/evaluation/daily_action_design_failure_review.md
```

内容至少包括：

1. 为什么 daily continuous action 容易打满 season cap；
2. action safety 在当前设计中扮演了什么角色；
3. 为什么 reward cost 和 terminal profit reward 都没有改变打满行为；
4. 当前动作频率与真实农田管理的差异；
5. 为什么应该尝试 low-frequency 或 scheduled management action；
6. 本阶段建议优先实现哪些最小侵入方案。

---

## 8. 第二步：设计 action design candidates

请至少设计 3 种 action design。先写入：

```text
Leave_One_experiments/action_design_debug/evaluation/action_design_candidate_plan.md
```

### Design A: decision interval wrapper

低频决策 wrapper。PPO 不是每天决策，而是每隔 N 天决策一次；非决策日动作强制为 0。

建议测试：

```text
A7: decision_interval_days = 7
A10: decision_interval_days = 10
A15: decision_interval_days = 15
```

### Design B: phenology window gated action

生育期窗口门控。只允许在特定 DAP 窗口施肥/灌溉，其余时间强制为 0。

建议窗口：

```text
fertilization_windows:
  basal: DAP 1-7
  jointing_or_early_growth: DAP 25-40
  pre_tasseling_or_mid_growth: DAP 55-70

irrigation_windows:
  early_growth: DAP 20-35
  mid_growth: DAP 45-65
  silking_grainfill: DAP 70-95
```

### Design C: seasonal budget allocation action

预算分配式动作。PPO 不再每天直接决定 amir/anfer，而是决定几个阶段的水氮预算分配比例。

示例：

```text
irrigation_budget_total <= 300 mm
nitrogen_budget_total <= 450 kg/ha
stage fractions sum to <= 1
```

本阶段优先实现 Design A 和 Design B。Design C 先生成技术方案，如实现成本可控，再做最小版本。

---

## 9. 第三步：实现最小侵入 action wrappers

请新增：

```text
src/action_design_wrappers.py
```

至少实现：

```text
DecisionIntervalActionWrapper
PhenologyWindowActionWrapper
```

要求：

1. 不修改 gym-DSSAT 原始源码；
2. 不修改 my_data；
3. wrappers 能接在 SafeActionWrapper 之前或之后，但必须在报告中说明顺序；
4. 每日 CSV 必须记录 raw action 和 executed action；
5. 非决策日或非窗口日被强制为 0 的原因必须记录；
6. action safety 仍作为最后安全阀；
7. 不要破坏已有 PPO 训练/评估脚本。

建议记录字段：

```text
raw_real_action_amir
raw_real_action_anfer
design_filtered_action_amir
design_filtered_action_anfer
safe_real_action_amir
safe_real_action_anfer
action_design_rule_triggered
is_decision_day
is_in_fertilization_window
is_in_irrigation_window
```

---

## 10. 第四步：HLA pilot 设置

只先做 HLA。

```text
station = HLA
train_year = 2011
eval_years = 2011, 2007, 2009
seed = 0
timesteps = 5000
action_safety_enabled = true
season_irrigation_cap = 300 mm
season_n_cap = 450 kg/ha
daily_irrigation_max = 40 mm
daily_n_max = 80 kg/ha
```

reward 使用：

```text
episode-level profit reward wrapper
```

如果 episode-level profit reward 在当前脚本中不稳定，则使用当前 reward，但报告必须说明。

---

## 11. action design candidates to test

先测试：

```text
baseline_daily_action_current
A7_decision_interval_7d
A10_decision_interval_10d
A15_decision_interval_15d
B_window_gated_default
A10_plus_B_window_gated
```

说明：

1. baseline_daily_action_current 是对照；
2. A 系列检验降低决策频率是否有效；
3. B 检验农学窗口门控是否有效；
4. A10+B 检验低频决策和窗口门控叠加是否有效。

如果计算量太大，先运行：

```text
baseline_daily_action_current
A10_decision_interval_10d
B_window_gated_default
A10_plus_B_window_gated
```

---

## 12. 每个 candidate 训练前 smoke check

每个 action design candidate 训练前，必须运行：

```text
HLA 2011 null_zero
HLA 2011 fixed_low_input
```

保存到：

```text
Leave_One_experiments/action_design_debug/smoke_checks/
```

汇总表：

```text
Leave_One_experiments/action_design_debug/smoke_checks/pretrain_smoke_check_summary.csv
```

---

## 13. 每个 candidate 训练后评估

每个 action design candidate 完成训练后，必须评估：

```text
HLA eval 2011
HLA eval 2007
HLA eval 2009
```

输出 daily CSV 到：

```text
Leave_One_experiments/action_design_debug/daily_outputs/HLA/
```

输出 summary 到：

```text
Leave_One_experiments/action_design_debug/evaluation/action_design_evaluation_summary.csv
```

字段至少包括：

```text
station
action_design
reward_version
train_year
eval_year
seed
run_status
episode_completed
final_grnwt
total_irrigation
total_n_fertilizer
irrigation_saturation_ratio
n_saturation_ratio
mean_swfac
mean_nstres
mean_reward
sum_reward
num_decision_days
num_irrigation_allowed_days
num_fertilization_allowed_days
num_design_filtered_days
num_safety_trigger_days
daily_csv_path
figure_dir
notes
```

---

## 14. 判断标准

每个 action design candidate 计算：

```text
mean_yield
std_yield
mean_irrigation
mean_n
mean_irrigation_saturation_ratio
mean_n_saturation_ratio
yield_loss_vs_baseline
input_reduction_vs_baseline
mean_swfac
mean_nstres
```

输出：

```text
Leave_One_experiments/action_design_debug/evaluation/action_design_candidate_comparison.csv
```

初筛标准：

```text
mean_irrigation_saturation_ratio < 0.95
mean_n_saturation_ratio < 0.95
yield_loss_vs_baseline <= 15%
all episodes ok
```

宽松标准：

```text
至少一个投入维度 saturation_ratio < 0.95
yield_loss_vs_baseline <= 15%
all episodes ok
```

如果 action design 只是把投入强行压到 0，导致产量严重下降，则标记：

```text
action_too_restrictive
```

---

## 15. 图表输出

每个 candidate + eval_year 至少生成：

```text
daily_actions_raw_filtered_safe.png
cumulative_water_nitrogen.png
crop_growth_timeseries.png
action_design_triggers.png
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
```

汇总图至少包括：

```text
action_design_yield_vs_input.png
action_design_saturation_ratio.png
action_design_yield_loss_vs_input_reduction.png
action_design_filtered_days.png
action_design_swfac_nstres.png
```

保存到：

```text
Leave_One_experiments/action_design_debug/figures/
```

---

## 16. 如果 HLA 找到有效 action design，扩展到 SYA/LCA 小测试

如果 HLA 找到至少一个 action design 满足初筛或宽松标准，选择最优 action design 扩展到：

```text
SYA train_year = 2012
eval_years = 2012, 2014, 2015

LCA train_year = 2010
eval_years = 2010, 2011, 2008, 2009
```

输出：

```text
Leave_One_experiments/action_design_debug/evaluation/cross_site_action_design_test.csv
```

如果 HLA 没有任何有效 action design，不要扩展 SYA/LCA。

---

## 17. 如果 Design A/B 均失败

如果 Design A/B 仍然全部打满 cap，或全部导致产量崩塌，请不要继续乱调。

生成：

```text
Leave_One_experiments/action_design_debug/evaluation/action_design_blocker_report.md
```

说明下一步是否需要转向：

```text
explicit seasonal budget action
scheduled discrete fertilization/irrigation events
rule-based expert policy + imitation learning
hybrid PPO over management windows
```

---

## 18. 报告要求

报告必须说明：

1. 为什么 reward 调试失败后需要改 action design；
2. 当前 daily continuous action 的问题；
3. Design A/B/C 的区别；
4. 实现了哪些 wrapper；
5. wrapper 顺序；
6. HLA pilot 结果；
7. 哪些 action design 不再打满 cap；
8. 产量损失是否可接受；
9. SYA/LCA 小扩展结果；
10. 是否可以进入 multi-seed；
11. 是否可以进入 rainfall-scaling budget scenario；
12. 如果仍失败，下一步建议。

---

## 19. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_action_design_debug_report.md
```

生成 PPT：

```text
docs/2026-06-06_action_design_debug_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/action_design_debug/reports/
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
git add prompts/006_06_revise_action_design_scheduled_discrete_management.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/action_design_debug/configs/
git add Leave_One_experiments/action_design_debug/evaluation/
git add Leave_One_experiments/action_design_debug/figures/
git add Leave_One_experiments/action_design_debug/reports/
git add docs/2026-06-06_action_design_debug_report.md
git add docs/2026-06-06_action_design_debug_report.pptx
git commit -m "Revise action design for scheduled discrete management"
```

注意：

1. 不要默认 commit 大模型 `.zip`；
2. 不要默认 commit tensorboard 大日志；
3. 不要默认 commit 过大的 daily_outputs；
4. 不要强行 push。

---

## 21. 完成后请汇报

完成后请汇报：

1. 哪些 action design 被实现；
2. wrapper 顺序是什么；
3. HLA 是否有 action design 不再打满 300/450；
4. 产量损失是否 <= 15%；
5. 推荐 action design 是哪个；
6. SYA/LCA 小扩展是否执行；
7. SYA/LCA 是否也不再打满 cap；
8. 是否可以进入多 seed；
9. 是否可以进入 rainfall-scaling budget scenario；
10. 如果仍失败，是否需要 explicit seasonal budget action。
