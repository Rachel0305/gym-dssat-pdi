# 006_07_explicit_seasonal_budget_or_scheduled_event_action

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_05_redesign_episode_level_profit_reward.md
prompts/006_06_revise_action_design_scheduled_discrete_management.md
docs/2026-06-06_episode_profit_reward_debug_report.md
docs/2026-06-06_action_design_debug_report.md
```

现在执行新的子任务：从每日连续动作 / 低频动作 / 窗口门控，进一步转向显式季节预算动作或固定农事事件动作，先做 HLA 2011 pilot。

本阶段不要做多 seed，不要进入 rainfall-scaling budget scenario，不要训练五站点，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

前面多个阶段已经证明，当前 PPO 的问题不是单纯 reward 系数问题。

006_03、006_04、006_05 都证明，即使加入 step-wise cost、强成本、terminal profit reward，HLA pilot 仍然打满：

```text
300 mm irrigation / 450 kg ha-1 N
```

006_06 已经实现并测试：

```text
ScheduledActionDesignWrapper
DecisionIntervalActionWrapper
PhenologyWindowActionWrapper
```

wrapper 顺序为：

```text
GymDssatWrapper -> ActionDesignWrapper -> SafeActionWrapper -> EpisodeProfitRewardWrapper
```

但是 HLA pilot 中以下 action design 仍然全部打满 300/450：

```text
baseline_daily_action_current
A7_decision_interval_7d
A10_decision_interval_10d
A15_decision_interval_15d
B_window_gated_default
A10_plus_B_window_gated
```

没有任何 action design 通过 strict 或 relaxed filter。

因此，继续做“每日 action 的过滤/降频”已经不够。下一步应转向更强的 action design：

```text
explicit seasonal budget action
scheduled discrete event action
stage-level budget allocation action
```

---

## 2. 当前结论

当前结果说明：

1. PPO 每日连续动作会打满 cap；
2. step-wise reward 成本项无法阻止；
3. terminal profit reward 也无法阻止；
4. 低频决策 wrapper 仍无法阻止；
5. 生育期窗口门控仍无法阻止；
6. action safety 仍然只是最后安全阀，不是策略本身；
7. PPO 需要从“每天决定施多少”改成“先决定季节/阶段预算，再按规则执行”。

---

## 3. 本阶段目标

1. 设计更强的 action design；
2. 优先实现显式季节预算动作或固定农事事件动作；
3. 不再让 PPO 每天直接输出水氮量；
4. 只在 HLA 2011 做 pilot；
5. 检查 PPO 是否不再打满 300/450；
6. 检查产量损失是否可接受；
7. 如果 HLA 成功，再扩展 SYA/LCA 小测试；
8. 如果仍失败，判断是否应该转为 rule-based expert policy + imitation learning，或者先做非 RL 的预算情景优化。

---

## 4. 本阶段不要做的事情

1. 不要做多 seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练 FQA/YCA。
4. 不要训练五站点。
5. 不要训练无 action safety 的 PPO。
6. 不要修改 `my_data/` 原始文件。
7. 不要覆盖 site-packages 中的原始 reward 文件。
8. 不要覆盖 006_03、006_04、006_05、006_06 结果。
9. 不要继续只做 daily action filter。
10. 不要大改 DSSAT Fortran 源码。
11. 不要把本阶段结果写成最终论文结论。

---

## 5. 输入文件

优先读取：

```text
docs/2026-06-06_action_design_debug_report.md
Leave_One_experiments/action_design_debug/evaluation/action_design_evaluation_summary.csv
Leave_One_experiments/action_design_debug/evaluation/action_design_candidate_comparison.csv
src/action_design_wrappers.py
src/episode_profit_reward.py
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
sb3_wrapper.py
experiments/ppo_observed_years/config_season_cap_sensitivity.yaml
experiments/ppo_observed_years/config_ppo_action_safe_site_training.yaml
```

如果路径不同，请搜索文件名，不要猜。

---

## 6. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/budget_action_design_debug/
```

建议目录结构：

```text
Leave_One_experiments/budget_action_design_debug/
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
docs/2026-06-06_budget_action_design_debug_report.md
docs/2026-06-06_budget_action_design_debug_report.pptx
```

---

## 7. 第一步：复盘 A/B action design 为什么失败

请生成：

```text
Leave_One_experiments/budget_action_design_debug/evaluation/action_design_failure_review.md
```

内容至少包括：

1. A7/A10/A15 是否减少了实际投入；
2. B window gated 是否减少了实际投入；
3. 为什么 mean_irrigation 和 mean_n 仍然是 300/450；
4. safety trigger days 是否仍然很高；
5. decision days 减少后为什么仍然打满 cap；
6. 当前 wrappers 是否只是过滤每日动作，而没有改变“总预算决策”；
7. 为什么下一步需要 explicit budget/action event design。

---

## 8. 第二步：设计更强 action design candidates

请生成：

```text
Leave_One_experiments/budget_action_design_debug/evaluation/budget_action_design_plan.md
```

至少设计以下 3 类。

### Design D: explicit seasonal budget action

PPO 不再每天输出 `amir/anfer`，而是在 episode 开始或少数决策点输出：

```text
season_irrigation_budget
season_n_budget
```

然后由规则把预算分配到固定生育期窗口。

建议预算动作范围：

```text
season_irrigation_budget: 0-300 mm
season_n_budget: 0-450 kg/ha
```

建议固定分配比例：

```text
irrigation:
  early_growth: 20%
  mid_growth: 40%
  silking_grainfill: 40%

nitrogen:
  basal: 30%
  early_growth: 30%
  pre_tasseling: 40%
```

### Design E: scheduled discrete event amounts

PPO 只在固定事件日输出事件施用量，其余日期强制为 0。

建议事件：

```text
fertilization_events:
  DAP 1
  DAP 30
  DAP 60

irrigation_events:
  DAP 25
  DAP 50
  DAP 75
```

每次事件 action 上限：

```text
single_irrigation_event_max = 100 mm
single_n_event_max = 150 kg/ha
```

### Design F: stage-level budget allocation

PPO 在阶段开始时输出该阶段预算，阶段内由规则执行。

建议阶段：

```text
stage_1: DAP 1-30
stage_2: DAP 31-60
stage_3: DAP 61-95
stage_4: DAP 96-120
```

每阶段动作：

```text
stage_irrigation_budget
stage_n_budget
```

本阶段优先实现 Design D 和 Design E。Design F 可以先生成技术方案，如果实现成本不高再做最小版本。

---

## 9. 第三步：实现最小侵入 wrappers

请新增或更新：

```text
src/budget_action_wrappers.py
```

至少实现：

```text
SeasonalBudgetActionWrapper
ScheduledEventActionWrapper
```

如果可行，再实现：

```text
StageBudgetActionWrapper
```

要求：

1. 不修改 gym-DSSAT 原始源码；
2. 不修改 my_data；
3. wrapper 应在 SafeActionWrapper 前执行；
4. SafeActionWrapper 仍作为最后安全阀；
5. EpisodeProfitRewardWrapper 仍在最后计算 terminal profit；
6. daily CSV 必须记录 PPO raw action、budget/action event、最终执行 action；
7. 所有自动分配规则必须记录到配置文件；
8. 不允许静默把 action 改成 0。

建议 wrapper 顺序：

```text
GymDssatWrapper
-> BudgetActionWrapper
-> SafeActionWrapper
-> EpisodeProfitRewardWrapper
```

---

## 10. daily CSV 新增字段

至少包含：

```text
raw_policy_action
budget_action_irrigation
budget_action_n
scheduled_event_irrigation
scheduled_event_n
executed_irrigation
executed_n
budget_remaining_irrigation
budget_remaining_n
stage_name
event_name
is_budget_decision_day
is_scheduled_event_day
budget_rule_triggered
safe_rule_triggered
```

如果字段命名与现有框架不同，请在报告中说明映射关系。

---

## 11. HLA pilot 设置

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
daily_irrigation_max = 100 mm
daily_n_max = 150 kg/ha
```

注意：

这里 daily max 可以比之前更高，因为事件数量少，但 season cap 仍然作为安全阀。

reward 使用：

```text
episode-level profit reward wrapper
```

如果 episode-level profit reward 在当前脚本中不稳定，则使用最可解释 reward，但报告必须说明。

---

## 12. action design candidates to test

先测试：

```text
baseline_daily_action_current
D_seasonal_budget_fixed_split
E_scheduled_events_default
F_stage_budget_default
D_plus_profit_reward
E_plus_profit_reward
```

说明：

1. baseline_daily_action_current 是对照；
2. D 检验显式季节预算是否有效；
3. E 检验固定事件动作是否有效；
4. F 检验阶段预算是否有效；
5. D/E plus profit reward 检查预算动作和 episode profit 是否能一起发挥作用。

如果计算量太大，先运行：

```text
baseline_daily_action_current
D_seasonal_budget_fixed_split
E_scheduled_events_default
D_plus_profit_reward
E_plus_profit_reward
```

---

## 13. 每个 candidate 训练前 smoke check

每个 candidate 训练前，必须运行：

```text
HLA 2011 null_zero
HLA 2011 fixed_low_input
```

保存到：

```text
Leave_One_experiments/budget_action_design_debug/smoke_checks/
```

汇总表：

```text
Leave_One_experiments/budget_action_design_debug/smoke_checks/pretrain_smoke_check_summary.csv
```

---

## 14. 每个 candidate 训练后评估

每个 candidate 完成训练后，必须评估：

```text
HLA eval 2011
HLA eval 2007
HLA eval 2009
```

输出 daily CSV 到：

```text
Leave_One_experiments/budget_action_design_debug/daily_outputs/HLA/
```

输出 summary 到：

```text
Leave_One_experiments/budget_action_design_debug/evaluation/budget_action_design_evaluation_summary.csv
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
num_budget_decision_days
num_scheduled_event_days
num_safe_trigger_days
daily_csv_path
figure_dir
notes
```

---

## 15. 判断标准

每个 candidate 计算：

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
Leave_One_experiments/budget_action_design_debug/evaluation/budget_action_design_candidate_comparison.csv
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

## 16. 图表输出

每个 candidate + eval_year 至少生成：

```text
budget_or_event_actions.png
cumulative_water_nitrogen.png
crop_growth_timeseries.png
budget_remaining_timeseries.png
safe_trigger_timeseries.png
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
```

汇总图至少包括：

```text
budget_action_yield_vs_input.png
budget_action_saturation_ratio.png
budget_action_yield_loss_vs_input_reduction.png
budget_action_safe_trigger_days.png
budget_action_swfac_nstres.png
```

保存到：

```text
Leave_One_experiments/budget_action_design_debug/figures/
```

---

## 17. 如果 HLA 找到有效 design，扩展到 SYA/LCA 小测试

如果 HLA 找到至少一个 design 满足初筛或宽松标准，选择最优 design 扩展到：

```text
SYA train_year = 2012
eval_years = 2012, 2014, 2015

LCA train_year = 2010
eval_years = 2010, 2011, 2008, 2009
```

输出：

```text
Leave_One_experiments/budget_action_design_debug/evaluation/cross_site_budget_action_design_test.csv
```

如果 HLA 没有任何有效 design，不要扩展 SYA/LCA。

---

## 18. 如果 Design D/E/F 均失败

如果 D/E/F 仍然全部打满 cap，或全部导致产量崩塌，请不要继续乱调。

生成：

```text
Leave_One_experiments/budget_action_design_debug/evaluation/budget_action_design_blocker_report.md
```

说明下一步是否需要转向：

```text
rule-based expert policy + imitation learning
offline search over fixed management schedules
Bayesian optimization over water/N budgets
DSSAT scenario ensemble rather than PPO
```

---

## 19. 报告要求

报告必须说明：

1. 为什么 A/B action design 失败后需要 explicit budget/event action；
2. D/E/F 的区别；
3. 实现了哪些 wrappers；
4. wrapper 顺序；
5. HLA pilot 结果；
6. 哪些 design 不再打满 cap；
7. 产量损失是否可接受；
8. SYA/LCA 小扩展结果；
9. 是否可以进入 multi-seed；
10. 是否可以进入 rainfall-scaling budget scenario；
11. 如果仍失败，下一步建议。

---

## 20. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_budget_action_design_debug_report.md
```

生成 PPT：

```text
docs/2026-06-06_budget_action_design_debug_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/budget_action_design_debug/reports/
```

---

## 21. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/006_07_explicit_seasonal_budget_or_scheduled_event_action.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/budget_action_design_debug/configs/
git add Leave_One_experiments/budget_action_design_debug/evaluation/
git add Leave_One_experiments/budget_action_design_debug/figures/
git add Leave_One_experiments/budget_action_design_debug/reports/
git add docs/2026-06-06_budget_action_design_debug_report.md
git add docs/2026-06-06_budget_action_design_debug_report.pptx
git commit -m "Implement explicit budget and scheduled event action designs"
```

注意：

1. 不要默认 commit 大模型 `.zip`；
2. 不要默认 commit tensorboard 大日志；
3. 不要默认 commit 过大的 daily_outputs；
4. 不要强行 push。

---

## 22. 完成后请汇报

完成后请汇报：

1. 哪些 budget/event action design 被实现；
2. wrapper 顺序是什么；
3. HLA 是否有 design 不再打满 300/450；
4. 产量损失是否 <= 15%；
5. 推荐 design 是哪个；
6. SYA/LCA 小扩展是否执行；
7. SYA/LCA 是否也不再打满 cap；
8. 是否可以进入多 seed；
9. 是否可以进入 rainfall-scaling budget scenario；
10. 如果仍失败，是否应转向 imitation learning 或非 RL 预算优化。
