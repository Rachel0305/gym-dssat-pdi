# 006_08_offline_schedule_search_and_bayesian_optimization_prior

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_05_redesign_episode_level_profit_reward.md
prompts/006_06_revise_action_design_scheduled_discrete_management.md
prompts/006_07_explicit_seasonal_budget_or_scheduled_event_action.md
docs/2026-06-06_episode_profit_reward_debug_report.md
docs/2026-06-06_action_design_debug_report.md
docs/2026-06-06_budget_action_design_debug_report.md
```

如果文件名或日期略有不同，请搜索关键词：

```text
episode_profit_reward_debug_report
action_design_debug_report
budget_action_design_debug_report
```

现在执行新的子任务：暂停继续扩展 PPO，先用离线 schedule search / Bayesian optimization / DSSAT scenario ensemble 找到合理的水氮管理先验，为后续 imitation learning 或 constrained PPO 提供目标策略。

本阶段不要做多 seed，不要进入 rainfall-scaling budget scenario，不要继续训练新的 PPO 策略，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

前面多个阶段已经证明，当前 PPO 路线存在系统性问题：

### 006_03 reward cost debug

A/B/C reward candidates 全部打满：

```text
300 mm irrigation / 450 kg ha-1 N
```

### 006_04 reward coefficient tuning

D/E/F/G 更强成本 candidates 仍然全部打满：

```text
300 mm irrigation / 450 kg ha-1 N
```

### 006_05 episode-level profit reward

terminal profit reward 已经可在 wrapper 层实现，但 P0-P6 仍然全部打满：

```text
300 mm irrigation / 450 kg ha-1 N
```

### 006_06 low-frequency / phenology-window action design

A7/A10/A15、B_window_gated、A10+B 仍然全部打满：

```text
300 mm irrigation / 450 kg ha-1 N
```

### 006_07 explicit budget / scheduled event action design

已经实现：

```text
SeasonalBudgetActionWrapper
ScheduledEventActionWrapper
StageBudgetActionWrapper
```

wrapper 顺序为：

```text
GymDssatWrapper -> BudgetActionWrapper -> SafeActionWrapper -> EpisodeProfitRewardWrapper
```

但是 HLA pilot 结果仍然显示所有 design 都选择上限投入：

```text
D_seasonal_budget_fixed_split
D_plus_profit_reward
E_scheduled_events_default
E_plus_profit_reward
F_stage_budget_default
baseline_daily_action_current
```

全部：

```text
mean_irrigation = 300 mm
mean_n = 450 kg/ha
saturation_ratio = 1.0
```

没有任何 design 通过 strict 或 relaxed filter，SYA/LCA extension 被跳过。

因此，继续对 PPO 的 reward 或 action wrapper 做小修小补已经不合适。下一步应该转向：

```text
offline schedule search
Bayesian optimization
DSSAT scenario ensemble
rule-based expert policy construction
imitation learning prior
```

---

## 2. 本阶段核心判断

当前结论是：

1. PPO 在当前观测变量、动作设计和 reward 下，总是倾向选择最大水氮；
2. 单纯加 action safety 只能防止爆炸，不能产生合理策略；
3. 单纯调 reward 系数无法改变行为；
4. terminal profit reward 也无法改变行为；
5. 低频、窗口、预算动作语义仍然无法让 PPO 主动节约；
6. 需要先通过非 RL 方法找出“合理管理策略长什么样”；
7. 再把这个策略作为 expert prior，用于 imitation learning、behavior cloning、reward calibration 或 constrained PPO。

---

## 3. 本阶段目标

本阶段目标是：

1. 暂停 PPO 策略扩展；
2. 对 HLA 2011 先做离线水氮 schedule 搜索；
3. 找出若干组高产但不过量的水氮方案；
4. 明确产量-水氮投入 trade-off；
5. 找出 Pareto frontier；
6. 判断是否真的需要 300/450 才能达到当前产量；
7. 构建 expert policy / management prior；
8. 如果 HLA 成功，再扩展 SYA/LCA；
9. 为后续 imitation learning 或 constrained PPO 提供可解释目标。

---

## 4. 本阶段不要做的事情

1. 不要做 PPO multi-seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练新的 PPO 策略。
4. 不要训练无 action safety 的 PPO。
5. 不要训练五站点。
6. 不要修改 `my_data/` 原始文件。
7. 不要覆盖 006_03 到 006_07 的结果。
8. 不要继续只改 reward 系数。
9. 不要继续只改 PPO wrapper。
10. 不要把 offline search 结果直接写成最终 RL 结果。
11. 不要忽略产量-水氮投入 trade-off。
12. 不要只看最高产量，必须同时看 water/N/profit/Pareto。

---

## 5. 输入文件

优先读取：

```text
docs/2026-06-06_budget_action_design_debug_report.md
Leave_One_experiments/budget_action_design_debug/evaluation/budget_action_design_evaluation_summary.csv
Leave_One_experiments/budget_action_design_debug/evaluation/budget_action_design_candidate_comparison.csv
src/budget_action_wrappers.py
src/episode_profit_reward.py
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
sb3_wrapper.py
```

同时读取作物输入和年份配置：

```text
weather_clean_qc/
Leave_One_experiments/wth_generated_qc/
data/observed_phenology_dates_standardized.csv
Leave_One_experiments/year_classification/observed_phenology_rainfall_rank_by_station.csv
experiments/ppo_observed_years/
my_data/
```

如果路径不同，请搜索文件名，不要猜。

---

## 6. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/offline_schedule_search/
```

建议目录结构：

```text
Leave_One_experiments/offline_schedule_search/
  configs/
  rendered_inputs/
  candidate_schedules/
  daily_outputs/
  evaluation/
  figures/
  expert_policy/
  reports/
```

报告保存到：

```text
docs/2026-06-06_offline_schedule_search_report.md
docs/2026-06-06_offline_schedule_search_report.pptx
```

如果日期不方便自动获取，可以使用当前系统日期。

---

## 7. 第一步：复盘为什么 PPO 方向暂时停止

请生成：

```text
Leave_One_experiments/offline_schedule_search/evaluation/why_pause_ppo_and_search_offline.md
```

内容至少包括：

1. 006_03 到 006_07 的主要结果；
2. 为什么 reward cost 不足；
3. 为什么 terminal profit reward 不足；
4. 为什么 low-frequency/window/budget action wrapper 仍然不足；
5. 为什么需要先用非 RL 方法找 expert prior；
6. 离线搜索结果如何服务后续 imitation learning / constrained PPO。

---

## 8. 第二步：定义 HLA 离线 schedule 搜索空间

先只做 HLA 2011。

```text
station = HLA
train_year = 2011
eval_years = 2011, 2007, 2009
```

先固定农事事件日，搜索每次事件水氮量。

建议事件日：

```text
nitrogen_events:
  N1: DAP 1
  N2: DAP 30
  N3: DAP 60

irrigation_events:
  I1: DAP 25
  I2: DAP 50
  I3: DAP 75
```

搜索范围：

```text
N1, N2, N3 each in [0, 50, 100, 150] kg/ha
I1, I2, I3 each in [0, 50, 100] mm
```

总上限：

```text
total_N <= 300 kg/ha for conservative search
total_irrigation <= 250 mm for conservative search
```

同时保留一个 high-input reference：

```text
total_N <= 450 kg/ha
total_irrigation <= 300 mm
```

---

## 9. 第三步：先做 coarse grid search

先做粗网格，不要一开始 Bayesian optimization。

候选数量估算：

```text
N combinations: 4^3 = 64
I combinations: 3^3 = 27
total = 1728
```

如果 1728 太多，先缩小为：

```text
N1,N2,N3 each in [0, 75, 150]
I1,I2,I3 each in [0, 50, 100]
total = 729
```

如果仍然太多，先做 HLA 2011 单年 729 个候选，再选择 top candidates 跨年验证。

每个候选 schedule 需要记录：

```text
schedule_id
station
train_year
N1_DAP
N1_amount
N2_DAP
N2_amount
N3_DAP
N3_amount
I1_DAP
I1_amount
I2_DAP
I2_amount
I3_DAP
I3_amount
total_N
total_irrigation
```

保存到：

```text
Leave_One_experiments/offline_schedule_search/candidate_schedules/HLA_2011_coarse_grid_schedules.csv
```

---

## 10. 第四步：运行 DSSAT/gym-DSSAT deterministic evaluation

对每个候选 schedule，运行 deterministic policy，不训练 PPO。

需要实现或复用：

```text
src/offline_schedule_policy.py
src/run_offline_schedule_search.py
```

policy 行为：

```text
如果当前 DAP 等于事件 DAP:
  执行对应 irrigation 或 nitrogen amount
否则:
  action = 0
```

必须保存每个候选的 daily output 和 summary。

summary 保存到：

```text
Leave_One_experiments/offline_schedule_search/evaluation/HLA_2011_coarse_grid_summary.csv
```

字段至少包括：

```text
schedule_id
station
eval_year
run_status
episode_completed
final_grnwt
final_topwt
final_xlai
total_irrigation
total_n_fertilizer
mean_swfac
mean_nstres
yield_per_100mm_irrigation
yield_per_100kg_n
profit_score
daily_csv_path
notes
```

profit_score 先使用 normalized 版本：

```text
profit_score = grain_value_coef * final_grnwt
               - water_cost * total_irrigation
               - n_cost * total_n_fertilizer
```

默认：

```text
grain_value_coef = 0.01
water_cost = 0.5
n_cost = 0.25
```

这些不是最终真实经济价格，只是筛选用 normalized score。

---

## 11. 第五步：筛选 Pareto frontier 和候选 expert schedules

从 HLA 2011 coarse grid 中筛选：

1. top yield schedules；
2. top profit schedules；
3. low input schedules；
4. Pareto-efficient schedules；
5. 与 PPO 300/450 baseline 相比，产量损失 <= 5%、10%、15% 且投入明显更低的 schedules。

输出：

```text
Leave_One_experiments/offline_schedule_search/evaluation/HLA_2011_pareto_frontier.csv
Leave_One_experiments/offline_schedule_search/evaluation/HLA_2011_expert_schedule_candidates.csv
```

筛选字段至少包括：

```text
schedule_id
final_grnwt
total_irrigation
total_n_fertilizer
profit_score
yield_loss_vs_ppo_baseline
irrigation_reduction_vs_ppo_baseline
n_reduction_vs_ppo_baseline
is_pareto
expert_candidate_type
```

---

## 12. 第六步：跨年份验证 top schedules

从 HLA 2011 中选择：

```text
top 5 yield
top 5 profit
top 5 Pareto-balanced
top 5 low-input-within-10%-yield-loss
```

去重后最多 20 个 schedules。

在 HLA 2007、2009、2011 上验证。

输出：

```text
Leave_One_experiments/offline_schedule_search/evaluation/HLA_top_schedule_cross_year_summary.csv
```

字段至少包括：

```text
schedule_id
eval_year
final_grnwt
total_irrigation
total_n_fertilizer
profit_score
mean_swfac
mean_nstres
run_status
episode_completed
```

再输出稳定性排名：

```text
Leave_One_experiments/offline_schedule_search/expert_policy/HLA_expert_schedule_ranking.csv
```

排名指标：

```text
mean_yield
std_yield
mean_profit
mean_irrigation
mean_n
yield_stability_score
profit_stability_score
overall_score
```

---

## 13. 第七步：如果 HLA 找到合理 expert schedule，扩展 SYA/LCA

如果 HLA 找到至少一个 schedule 满足：

```text
mean_yield_loss_vs_ppo_baseline <= 10%
mean_irrigation <= 250 mm
mean_n <= 300 kg/ha
run_status ok for all eval years
```

则把同样搜索流程扩展到：

```text
SYA train_year = 2012
eval_years = 2012, 2014, 2015

LCA train_year = 2010
eval_years = 2010, 2011, 2008, 2009
```

输出：

```text
Leave_One_experiments/offline_schedule_search/expert_policy/SYA_expert_schedule_ranking.csv
Leave_One_experiments/offline_schedule_search/expert_policy/LCA_expert_schedule_ranking.csv
```

如果 HLA 找不到合理 expert schedule，不要扩展 SYA/LCA。

---

## 14. 第八步：可选 Bayesian optimization

如果 coarse grid 太慢，或者 top schedules 不够好，可以在 HLA 上做 Bayesian optimization。

优化变量：

```text
N1, N2, N3
I1, I2, I3
```

边界：

```text
N_i: 0-150 kg/ha
I_i: 0-100 mm
```

约束：

```text
sum(N_i) <= 300
sum(I_i) <= 250
```

目标函数：

```text
maximize profit_score
```

或：

```text
maximize final_grnwt - lambda_water * total_irrigation - lambda_N * total_N
```

输出：

```text
Leave_One_experiments/offline_schedule_search/evaluation/HLA_bayesian_optimization_results.csv
```

注意：

Bayesian optimization 是可选项。如果 grid search 已经给出清晰 Pareto frontier，可以不做 BO。

---

## 15. 图表输出

HLA 至少生成：

```text
HLA_yield_vs_irrigation.png
HLA_yield_vs_nitrogen.png
HLA_profit_vs_input.png
HLA_pareto_frontier_yield_water_n.png
HLA_top_schedules_cross_year_yield.png
HLA_top_schedules_cross_year_profit.png
HLA_expert_schedule_actions.png
HLA_swfac_nstres_for_top_schedules.png
```

如果扩展 SYA/LCA，也生成同类图。

综合图：

```text
all_available_sites_expert_schedule_summary.png
```

保存到：

```text
Leave_One_experiments/offline_schedule_search/figures/
```

---

## 16. 后续用途：生成 imitation learning 数据

如果 HLA 或 SYA/LCA 找到 expert schedule，请导出 imitation learning 数据：

```text
Leave_One_experiments/offline_schedule_search/expert_policy/imitation_dataset.csv
```

字段至少包括：

```text
station
year
date
dap
state_variables
expert_action_irrigation
expert_action_n
schedule_id
expert_policy_type
```

说明：

这一步只是准备数据，不训练 imitation learning 模型。后续单独写 prompt。

---

## 17. 报告要求

报告必须说明：

1. 为什么暂停 PPO；
2. 为什么要做 offline schedule search；
3. 搜索空间；
4. deterministic policy 如何执行；
5. HLA coarse grid 结果；
6. Pareto frontier；
7. top schedules 跨年份表现；
8. 是否存在低投入但产量接近 PPO baseline 的策略；
9. 是否扩展 SYA/LCA；
10. 是否建议进入 imitation learning；
11. 是否建议回到 constrained PPO；
12. 是否建议做 rainfall-scaling budget scenario。

---

## 18. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_offline_schedule_search_report.md
```

生成 PPT：

```text
docs/2026-06-06_offline_schedule_search_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/offline_schedule_search/reports/
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
git add prompts/006_08_offline_schedule_search_and_bayesian_optimization_prior.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/offline_schedule_search/configs/
git add Leave_One_experiments/offline_schedule_search/evaluation/
git add Leave_One_experiments/offline_schedule_search/expert_policy/
git add Leave_One_experiments/offline_schedule_search/figures/
git add Leave_One_experiments/offline_schedule_search/reports/
git add docs/2026-06-06_offline_schedule_search_report.md
git add docs/2026-06-06_offline_schedule_search_report.pptx
git commit -m "Run offline schedule search for water nitrogen expert prior"
```

注意：

1. 不要默认 commit 大量 daily_outputs；
2. 不要默认 commit 大模型；
3. 不要默认 commit tensorboard；
4. 不要强行 push。

---

## 20. 完成后请汇报

完成后请汇报：

1. HLA coarse grid 是否完成；
2. 一共评估了多少 schedules；
3. 是否找到低投入且产量损失 <= 10% 的 schedule；
4. HLA best expert schedule 是哪个；
5. HLA top schedules 跨年份是否稳定；
6. 是否扩展到 SYA/LCA；
7. 是否生成 imitation_dataset.csv；
8. 是否建议下一步做 imitation learning；
9. 是否建议回到 constrained PPO；
10. 是否建议做 rainfall-scaling budget scenario。
