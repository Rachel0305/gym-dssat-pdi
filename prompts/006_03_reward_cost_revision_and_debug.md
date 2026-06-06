# 006_03_reward_cost_revision_and_debug

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/005_01_debug_ppo_action_scale_and_reward_before_batch_training.md
prompts/006_02_season_cap_sensitivity_analysis.md
docs/2026-06-06_season_cap_sensitivity_analysis_report.md
docs/2026-06-06_fqa_yca_two_year_action_safe_cv_report.md
docs/2026-06-06_action_safe_site_level_ppo_training_report.md
```

如果文件名或日期略有差异，请搜索以下关键词：

```text
season_cap_sensitivity_analysis_report
fqa_yca_two_year_action_safe_cv_report
action_safe_site_level_ppo_training_report
```

现在执行新的子任务：设计并调试带水氮成本项的 reward，使 PPO 不再仅仅依赖 action safety cap 才能避免过量水氮投入。

本阶段不要做多 seed，不要训练无 action safety 的 PPO，不要覆盖原 reward 文件。

---

## 1. 本阶段背景

上一阶段 `006_02_season_cap_sensitivity_analysis` 已经完成 season cap 敏感性分析。结果显示：

1. 计划模型数：25；
2. 完成模型数：25；
3. pretrain smoke checks：50/50 通过；
4. evaluations：70/70 通过；
5. 测试 cap：
   - 100 mm / 150 kg N
   - 150 mm / 225 kg N
   - 200 mm / 300 kg N
   - 250 mm / 375 kg N
   - 300 mm / 450 kg N
6. 所有站点、所有 cap、所有 evaluation 均打满对应 season cap：
   - `irrigation_ratio = 1.000`
   - `n_ratio = 1.000`
7. 报告结论明确指出：
   - PPO filled every tested season cap；
   - 当前 reward 仍然鼓励使用所有 action safety 允许的水氮；
   - 200/300 cap 只能视为 diagnostic safety setting，不是最终管理建议；
   - 多 seed 应该等 cap/cost design 不再被 safety ceiling 主导后再做。

因此，本阶段必须先修正 reward 的水氮成本逻辑，测试 PPO 是否能在较高 cap 下主动减少不必要水氮投入。

---

## 2. 本阶段核心问题

本阶段要回答：

1. 当前 reward 为什么会让 PPO 打满所有 cap？
2. 如果加入显式灌溉成本和施氮成本，PPO 是否还会打满 cap？
3. 成本项多大时，PPO 会开始减少水氮投入？
4. 减少水氮投入后，产量是否仍能保持合理？
5. reward 是否更符合“农学 / 经济学派”解释？
6. 是否可以确定一个后续多 seed 使用的 reward 版本？
7. 是否需要继续保留 action safety cap 作为安全阀？

---

## 3. 本阶段不要做的事情

1. 不要修改 `my_data/` 原始文件。
2. 不要直接覆盖 site-packages 中的原始 reward 文件。
3. 不要删除、覆盖前面 PPO、cap sensitivity、two-year CV 结果。
4. 不要做多 seed。
5. 不要训练无 action safety 的 PPO。
6. 不要直接使用无 safety PPO 作为有效策略。
7. 不要把本阶段 debug reward 直接写成最终论文方法。
8. 不要一次性跑五站点所有 reward 候选。
9. 不要只看 reward 数值，必须同时看产量、水氮投入、swfac、nstres。
10. 不要在没有报告说明的情况下自动替换正式 reward。

---

## 4. 本阶段输入

优先读取：

```text
docs/2026-06-06_season_cap_sensitivity_analysis_report.md
Leave_One_experiments/season_cap_sensitivity/evaluation/season_cap_sensitivity_evaluation_summary.csv
Leave_One_experiments/season_cap_sensitivity/evaluation/cap_saturation_summary.csv
Leave_One_experiments/season_cap_sensitivity/evaluation/marginal_response_by_cap.csv
Leave_One_experiments/ppo_action_safe_summary/all_site_best_policy_summary.csv
experiments/ppo_observed_years/config_season_cap_sensitivity.yaml
experiments/ppo_observed_years/config_ppo_action_safe_site_training.yaml
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
```

并定位当前 reward 函数实际来源。可能位置包括：

```text
gym_dssat_pdi/envs/configs/rewards.py
src/
experiments/
```

如果当前 reward 在 site-packages 中，不要直接覆盖；请复制候选 reward 到项目内受控路径。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/reward_cost_debug/
```

建议目录结构：

```text
Leave_One_experiments/reward_cost_debug/
  configs/
  reward_versions/
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
docs/2026-06-06_reward_cost_revision_and_debug_report.md
docs/2026-06-06_reward_cost_revision_and_debug_report.pptx
```

如果日期不方便自动获取，可以使用当前系统日期。

---

## 6. 第一步：复查当前 reward

请生成当前 reward 复查文件：

```text
Leave_One_experiments/reward_cost_debug/reward_versions/current_reward_review.md
```

内容至少包括：

1. 当前 reward 函数文件路径；
2. 当前 reward 函数名；
3. 完整代码片段；
4. 输入变量；
5. 使用了哪些作物状态变量；
6. 是否包含 daily irrigation cost；
7. 是否包含 daily nitrogen cost；
8. 是否包含 season irrigation cost；
9. 是否包含 season nitrogen cost；
10. 是否包含超过阈值后的非线性惩罚；
11. 是否包含终局产量收益；
12. 为什么当前 reward 会允许或鼓励打满 cap；
13. 和 cap sensitivity 结果的对应关系。

---

## 7. 第二步：设计 reward 候选版本

请不要直接覆盖原 reward。请把 reward 候选写到项目内：

```text
src/reward_candidates.py
```

或：

```text
Leave_One_experiments/reward_cost_debug/reward_versions/reward_candidates.py
```

同时生成说明文档：

```text
Leave_One_experiments/reward_cost_debug/reward_versions/reward_candidate_design.md
```

至少设计 3 类 reward 候选。

### 7.1 candidate_A_linear_cost

目标：最小修改，在现有 reward 上加入线性水氮成本。

形式示例：

```text
reward =
  crop_growth_reward
  - irrigation_cost_coef * daily_irrigation
  - nitrogen_cost_coef * daily_n
```

建议测试多个成本组合：

```text
A1: irrigation_cost_coef = 0.05, nitrogen_cost_coef = 0.02
A2: irrigation_cost_coef = 0.10, nitrogen_cost_coef = 0.05
A3: irrigation_cost_coef = 0.20, nitrogen_cost_coef = 0.10
```

### 7.2 candidate_B_linear_plus_excess_penalty

目标：在线性成本基础上，对接近或超过期望季节投入的行为加强惩罚。

形式示例：

```text
reward =
  crop_growth_reward
  - irrigation_cost_coef * daily_irrigation
  - nitrogen_cost_coef * daily_n
  - excess_irrigation_penalty
  - excess_n_penalty
```

建议期望阈值：

```text
target_irrigation = 200 mm
target_n = 300 kg/ha
```

但注意：这不是 hard cap，而是 reward penalty。

惩罚可以使用二次项：

```text
excess_irrigation_penalty = excess_irrigation_coef * max(0, cumulative_irrigation - target_irrigation)^2
excess_n_penalty = excess_n_coef * max(0, cumulative_n - target_n)^2
```

候选：

```text
B1: weak excess penalty
B2: medium excess penalty
B3: strong excess penalty
```

### 7.3 candidate_C_terminal_yield_minus_total_cost

目标：更接近经济学派解释，即“产量收益 - 水氮成本”。

形式示例：

```text
daily_reward = - daily_irrigation_cost - daily_nitrogen_cost
terminal_reward = yield_value_coef * final_grain_yield - total_irrigation_cost - total_nitrogen_cost
```

要求：

1. daily 期间主要惩罚投入；
2. episode 结束时给最终产量收益；
3. 明确单位和缩放；
4. 给出为什么这种设计更适合农学/经济学解释；
5. 先用于 debug，不直接替换正式 reward。

---

## 8. 第三步：建立 reward 候选测试计划

不要一开始五站点全跑。先做 HLA pilot。

测试站点和年份：

```text
station = HLA
train_year = 2011
eval_years = 2011, 2007, 2009
```

使用 cap：

```text
season_irrigation_cap = 300 mm
season_n_cap = 450 kg/ha
```

选择高 cap 的原因：

1. 如果 reward 有成本意识，PPO 不应自动打满 300/450；
2. 高 cap 更容易检测 reward 是否真的约束水氮；
3. action safety 仍作为安全阀，防止失控。

训练设置：

```text
seed = 0
total_timesteps = 5000
action_safety_enabled = true
daily_irrigation_max = 40
daily_n_max = 80
season_irrigation_soft_limit = 300
season_n_soft_limit = 450
```

先测试：

```text
current_reward_baseline
candidate_A1
candidate_A2
candidate_A3
candidate_B1
candidate_B2
candidate_B3
candidate_C1
```

如果计算量太大，先运行：

```text
current_reward_baseline
candidate_A2
candidate_B2
candidate_C1
```

---

## 9. 每个 reward 测试前 pretrain smoke check

每个 reward candidate 训练前，必须对：

```text
HLA 2011
null_zero
fixed_low_input
```

执行 pretrain smoke check。

保存到：

```text
Leave_One_experiments/reward_cost_debug/smoke_checks/
```

汇总：

```text
Leave_One_experiments/reward_cost_debug/smoke_checks/pretrain_smoke_check_summary.csv
```

如果 smoke check 失败，停止该 reward candidate。

---

## 10. 每个 reward 训练后评估

每个 reward candidate 完成训练后，必须评估：

```text
eval_years = 2011, 2007, 2009
```

每个 evaluation 保存 daily CSV、summary 和图。

daily output 保存到：

```text
Leave_One_experiments/reward_cost_debug/daily_outputs/HLA/
```

命名示例：

```text
HLA_train2011_candidate_A2_eval2007_seed0_daily.csv
```

summary 保存到：

```text
Leave_One_experiments/reward_cost_debug/evaluation/reward_candidate_evaluation_summary.csv
```

字段至少包括：

```text
station
reward_version
reward_family
train_year
eval_year
seed
cap_name
season_irrigation_cap
season_n_cap
model_path
run_status
episode_completed
final_grnwt
final_topwt
final_xlai
total_irrigation
total_n_fertilizer
irrigation_saturation_ratio
n_saturation_ratio
mean_swfac
mean_nstres
mean_reward
sum_reward
daily_csv_path
figure_dir
notes
```

---

## 11. reward 候选判断指标

每个 reward candidate 至少计算：

```text
mean_yield
std_yield
mean_reward
mean_irrigation
mean_n
mean_irrigation_saturation_ratio
mean_n_saturation_ratio
yield_loss_vs_current_reward
input_reduction_vs_current_reward
yield_per_100mm_irrigation
yield_per_100kg_n
```

输出：

```text
Leave_One_experiments/reward_cost_debug/evaluation/reward_candidate_comparison.csv
```

---

## 12. 推荐判定标准

推荐 reward candidate 不一定是产量最高，而是综合考虑：

1. 水氮不再总是打满高 cap；
2. 产量下降不能过大；
3. reward 数值稳定；
4. 跨 eval_year 表现稳定；
5. 投入减少有明显效果；
6. 农学/经济学解释清楚。

建议初步筛选条件：

```text
mean_irrigation_saturation_ratio < 0.95
mean_n_saturation_ratio < 0.95
yield_loss_vs_current_reward <= 10%
run_status = ok
episode_completed = True
```

如果所有候选仍打满 300/450 cap，说明成本项仍太弱，需要增强成本系数。

如果所有候选产量大幅下降，说明成本项太强或 reward scaling 不合理。

---

## 13. 图表输出

每个 reward candidate 至少生成：

```text
daily_actions_raw_vs_safe.png
cumulative_water_nitrogen.png
crop_growth_timeseries.png
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
```

每个 candidate + eval_year 保存到：

```text
Leave_One_experiments/reward_cost_debug/figures/HLA/{reward_version}/eval_{eval_year}/
```

汇总图保存到：

```text
Leave_One_experiments/reward_cost_debug/figures/summary/
```

至少包括：

```text
reward_candidates_yield_vs_input.png
reward_candidates_irrigation_n_ratio.png
reward_candidates_reward_vs_yield.png
reward_candidates_saturation_ratio.png
reward_candidates_cross_year_yield_stability.png
```

---

## 14. 如果 HLA pilot 成功，扩展到 SYA/LCA 小测试

如果 HLA pilot 找到了至少一个 candidate 满足：

```text
mean_irrigation_saturation_ratio < 0.95
mean_n_saturation_ratio < 0.95
yield_loss_vs_current_reward <= 10%
```

则用该 candidate 在 SYA 和 LCA 各做一个 best-policy train_year 测试：

```text
SYA train_year = 2012
eval_years = 2012, 2014, 2015

LCA train_year = 2010
eval_years = 2010, 2011, 2008, 2009
```

只测试 HLA 选出的最佳 reward candidate，不要把所有候选扩展到 SYA/LCA。

输出：

```text
Leave_One_experiments/reward_cost_debug/evaluation/cross_site_reward_candidate_test.csv
```

---

## 15. 报告解释要求

报告中必须明确说明：

1. action safety cap 是安全阀；
2. reward cost 是让 PPO 主动节约水氮；
3. 二者不是同一层机制；
4. 仅靠 cap 会导致 PPO 总是打满上限；
5. 加入成本项后，如果 PPO 不再打满高 cap，说明 reward 开始发挥约束作用；
6. 当前阶段仍是 debug，不是最终论文 reward；
7. 后续可根据真实水价、氮肥价格、产量价格，将 reward 改成经济学单位。

---

## 16. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_reward_cost_revision_and_debug_report.md
```

报告至少包括：

1. 为什么必须修 reward；
2. 当前 reward 复查；
3. season cap 敏感性结果如何证明 cap 主导；
4. reward 候选设计；
5. HLA pilot 设置；
6. 每个 reward candidate 结果；
7. 水氮投入是否下降；
8. 产量是否保持；
9. saturation ratio 是否降低；
10. 推荐 reward candidate；
11. SYA/LCA 小扩展测试结果；
12. 是否可以进入多 seed；
13. 是否可以进入 rainfall-scaling budget scenario；
14. 下一步建议。

---

## 17. PPT 输出

生成 PPT：

```text
docs/2026-06-06_reward_cost_revision_and_debug_report.pptx
```

PPT 至少包括：

1. 问题背景；
2. 为什么 cap 不够；
3. 当前 reward 问题；
4. reward candidate 设计；
5. HLA pilot 结果；
6. reward candidate 对水氮投入的影响；
7. reward candidate 对产量的影响；
8. 推荐 reward；
9. SYA/LCA 验证；
10. 下一步计划。

并复制一份到：

```text
Leave_One_experiments/reward_cost_debug/reports/
```

---

## 18. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/006_03_reward_cost_revision_and_debug.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/reward_cost_debug/configs/
git add Leave_One_experiments/reward_cost_debug/reward_versions/
git add Leave_One_experiments/reward_cost_debug/evaluation/
git add Leave_One_experiments/reward_cost_debug/figures/
git add Leave_One_experiments/reward_cost_debug/reports/
git add docs/2026-06-06_reward_cost_revision_and_debug_report.md
git add docs/2026-06-06_reward_cost_revision_and_debug_report.pptx
git commit -m "Debug reward cost terms for action-safe PPO"
```

注意：

1. 不要默认 commit 大模型 `.zip`；
2. 不要默认 commit tensorboard 大日志；
3. 不要默认 commit 过大的 daily_outputs；
4. 如果需要保存小型模型，请先报告文件大小；
5. 不要强行 push。

---

## 19. 完成后请汇报

完成后请汇报：

1. 当前 reward 的主要问题；
2. 测试了哪些 reward candidate；
3. 哪些 candidate 让水氮投入不再打满 300/450 cap；
4. 哪些 candidate 产量损失小于 10%；
5. 推荐使用哪个 candidate 进入下一阶段；
6. SYA/LCA 小扩展测试是否通过；
7. 是否可以进入多 seed 稳定性分析；
8. 是否可以进入 rainfall-scaling budget scenario；
9. 是否还需要继续调整成本系数。

---

## 20. 最重要原则

1. 本阶段是 reward 成本项 debug，不是最终论文 reward。
2. 不修改原始 reward 文件。
3. 不训练无 safety PPO。
4. 不做多 seed。
5. 不覆盖旧结果。
6. 只先用 HLA pilot。
7. HLA 找到有效 candidate 后，才扩展到 SYA/LCA。
8. 如果所有 candidate 仍打满 cap，继续增强成本项，不进入多 seed。
