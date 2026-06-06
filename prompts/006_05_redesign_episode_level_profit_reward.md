# 006_05_redesign_episode_level_profit_reward

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_03_reward_cost_revision_and_debug.md
prompts/006_04_tune_reward_cost_coefficients_until_unsaturated.md
docs/2026-06-06_reward_cost_revision_and_debug_report.md
docs/2026-06-06_reward_cost_coefficient_tuning_report.md
```

现在执行新的子任务：重构 reward 为 episode-level profit / season-level objective，而不是继续简单调 step-wise 成本系数。

本阶段不要做多 seed，不要进入 rainfall-scaling budget scenario，不要训练五站点，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

前两轮 reward debug 已经失败。

`006_03` 测试了 current_reward_baseline、candidate_A、candidate_B、candidate_C，所有候选都打满 300 mm 灌溉和 450 kg/ha 氮肥。

`006_04` 进一步测试了 candidate_D 强线性成本、candidate_E 累计二次成本、candidate_F 目标区间惩罚、candidate_G 经济学近似 reward。结果仍然是所有候选都打满 300/450，strict_pass、relaxed_pass、partial_pass 全部为 False，recommended candidate 为 None。

这说明继续简单调 step-wise reward 系数已经不足以解决问题。下一步应重构 reward 结构，让 PPO 优化一个清楚的季节目标：

```text
profit = final grain yield value - total irrigation cost - total nitrogen cost
```

而不是在每日 step 里零散地加成本。

---

## 2. 本阶段核心判断

本阶段要验证：

1. 是否能够在 episode 结束时明确读取 final_grnwt；
2. 是否能够在 episode 内可靠累计 total_irrigation 和 total_n；
3. 是否能够把 terminal profit reward 正确加到最后一个 step；
4. 使用 terminal profit reward 后，HLA 是否还会打满 300/450 cap；
5. 是否能找到 yield 损失可接受、水氮投入下降的 profit reward 参数；
6. 如果仍然打满 cap，是否说明 action design 或训练方式需要调整；
7. 如果水氮直接降到 0，是否说明成本过强或 yield reward 缩放过弱。

---

## 3. 本阶段不要做的事情

1. 不要做多 seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练 FQA/YCA。
4. 不要训练五站点。
5. 不要训练无 action safety 的 PPO。
6. 不要修改 `my_data/` 原始文件。
7. 不要覆盖 site-packages 中的原始 reward 文件。
8. 不要覆盖 `006_03` 或 `006_04` 结果。
9. 不要继续只靠简单调大 daily cost 系数。
10. 不要把本阶段 reward 直接写成最终论文 reward。
11. 不要只看 reward，必须同时看 yield、water、N、saturation ratio、swfac、nstres。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_reward_cost_coefficient_tuning_report.md
Leave_One_experiments/reward_cost_tuning/evaluation/reward_cost_tuning_evaluation_summary.csv
Leave_One_experiments/reward_cost_tuning/evaluation/reward_cost_tuning_candidate_comparison.csv
Leave_One_experiments/reward_cost_tuning/reward_versions/reward_candidates_v2.py
src/reward_candidates_v2.py
src/ppo_action_safety.py
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
sb3_wrapper.py
gym_dssat_pdi/envs/configs/rewards.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/episode_profit_reward_debug/
```

建议目录结构：

```text
Leave_One_experiments/episode_profit_reward_debug/
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
docs/2026-06-06_episode_profit_reward_debug_report.md
docs/2026-06-06_episode_profit_reward_debug_report.pptx
```

---

## 6. 第一步：检查环境能否支持 terminal reward

请先不要训练 PPO，先检查 reward 接口。

生成：

```text
Leave_One_experiments/episode_profit_reward_debug/evaluation/reward_interface_check.md
```

必须回答：

1. reward 函数是否能访问 `_next_state`；
2. reward 函数是否能判断 episode done；
3. reward 函数是否能访问 `_history`；
4. `_history` 是否包含 actions；
5. `_history` 是否包含 daily states；
6. 是否能从 `_next_state` 得到 final_grnwt；
7. 是否能累计 daily `amir` 和 `anfer`；
8. 是否能在最后一天加入 terminal reward；
9. 如果不能在原 reward 函数中判断 done，应该在哪一层加 terminal reward；
10. 需要改 `src/ppo_train.py`、wrapper，还是 reward function。

请优先采用最小侵入方案，不要大改 gym-DSSAT 源码。

---

## 7. 第二步：实现 episode-level reward wrapper

请新增或更新：

```text
src/episode_profit_reward.py
```

设计一个可配置的 reward wrapper 或 reward function，使其支持：

```text
daily_reward = - daily_water_cost * daily_irrigation - daily_n_cost * daily_n
terminal_reward = grain_value_coef * final_grnwt
                  - season_water_cost * total_irrigation
                  - season_n_cost * total_n
```

总思想：

```text
训练期间每天给投入成本惩罚；
episode 结束时一次性给最终产量收益和总投入成本；
最终优化目标接近 season profit。
```

如果当前框架不方便同时给 daily 和 terminal reward，则先实现 terminal-only 版本，但必须在报告中说明。

---

## 8. reward 版本设计

请设计以下 reward candidates。

```text
P0_current_reward_baseline

P1_terminal_profit_weak_cost:
  grain_value_coef = 0.01
  season_water_cost = 0.1
  season_n_cost = 0.05
  daily_water_cost = 0.0
  daily_n_cost = 0.0

P2_terminal_profit_medium_cost:
  grain_value_coef = 0.01
  season_water_cost = 0.5
  season_n_cost = 0.25
  daily_water_cost = 0.0
  daily_n_cost = 0.0

P3_terminal_profit_strong_cost:
  grain_value_coef = 0.01
  season_water_cost = 1.0
  season_n_cost = 0.5
  daily_water_cost = 0.0
  daily_n_cost = 0.0

P4_terminal_profit_with_daily_cost:
  grain_value_coef = 0.01
  season_water_cost = 0.5
  season_n_cost = 0.25
  daily_water_cost = 0.05
  daily_n_cost = 0.02

P5_lower_grain_value_medium_cost:
  grain_value_coef = 0.005
  season_water_cost = 0.5
  season_n_cost = 0.25
  daily_water_cost = 0.05
  daily_n_cost = 0.02

P6_high_economic_pressure:
  grain_value_coef = 0.005
  season_water_cost = 1.0
  season_n_cost = 0.5
  daily_water_cost = 0.1
  daily_n_cost = 0.05
```

说明：

1. 这些系数是 normalized score，不是真实人民币；
2. 目标是 debug reward 结构是否能改变 PPO 行为；
3. 后续可以映射到真实玉米价格、水价、氮肥价格。

---

## 9. HLA pilot 设置

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

先运行：

```text
P0_current_reward_baseline
P1
P2
P3
P4
P5
P6
```

如果计算量太大，先运行：

```text
P0
P2
P4
P6
```

---

## 10. 每个 reward 训练前 smoke check

每个 reward candidate 训练前，必须运行：

```text
HLA 2011 null_zero
HLA 2011 fixed_low_input
```

保存到：

```text
Leave_One_experiments/episode_profit_reward_debug/smoke_checks/
```

汇总表：

```text
Leave_One_experiments/episode_profit_reward_debug/smoke_checks/pretrain_smoke_check_summary.csv
```

---

## 11. 每个 reward 训练后评估

每个 reward candidate 完成训练后，必须评估：

```text
HLA eval 2011
HLA eval 2007
HLA eval 2009
```

输出 daily CSV 到：

```text
Leave_One_experiments/episode_profit_reward_debug/daily_outputs/HLA/
```

输出 summary 到：

```text
Leave_One_experiments/episode_profit_reward_debug/evaluation/episode_profit_reward_evaluation_summary.csv
```

字段至少包括：

```text
station
reward_version
reward_family
train_year
eval_year
seed
season_irrigation_cap
season_n_cap
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
terminal_reward
daily_cost_total
season_profit_score
daily_csv_path
figure_dir
notes
```

---

## 12. 判断标准

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
profit_score
mean_swfac
mean_nstres
```

输出：

```text
Leave_One_experiments/episode_profit_reward_debug/evaluation/episode_profit_reward_candidate_comparison.csv
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

如果某个 candidate 直接学成 0 水 0 氮且产量大幅下降，也要记录为 cost_too_strong。

---

## 13. 图表输出

每个 candidate + eval_year 至少生成：

```text
daily_actions_raw_vs_safe.png
cumulative_water_nitrogen.png
crop_growth_timeseries.png
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
terminal_profit_breakdown.png
```

汇总图至少包括：

```text
episode_profit_yield_vs_input.png
episode_profit_saturation_ratio.png
episode_profit_yield_loss_vs_input_reduction.png
episode_profit_profit_score_vs_yield.png
episode_profit_swfac_nstres.png
```

保存到：

```text
Leave_One_experiments/episode_profit_reward_debug/figures/
```

---

## 14. 如果 HLA 找到有效 candidate，扩展到 SYA/LCA 小测试

如果 HLA 找到至少一个 candidate 满足初筛或宽松标准，选择最优 candidate 扩展到：

```text
SYA train_year = 2012
eval_years = 2012, 2014, 2015

LCA train_year = 2010
eval_years = 2010, 2011, 2008, 2009
```

输出：

```text
Leave_One_experiments/episode_profit_reward_debug/evaluation/cross_site_episode_profit_reward_test.csv
```

如果 HLA 没有任何有效 candidate，不要扩展 SYA/LCA。

---

## 15. 如果 terminal reward 接口无法实现

如果发现当前 gym-DSSAT reward 接口不支持 terminal reward，不要硬改。

请生成：

```text
Leave_One_experiments/episode_profit_reward_debug/evaluation/terminal_reward_blocker_report.md
```

说明：

1. 阻塞点在哪个文件；
2. 为什么无法判断 episode done；
3. 需要修改哪个 wrapper；
4. 推荐最小修改方案；
5. 需要用户确认后再改。

---

## 16. 报告要求

报告必须说明：

1. 为什么 step-wise cost tuning 失败；
2. terminal reward 是否能实现；
3. episode-level profit reward 如何实现；
4. P0-P6 的结果；
5. 哪些 candidate 不再打满 cap；
6. 产量损失是否可接受；
7. 是否推荐某个 candidate；
8. SYA/LCA 小扩展结果；
9. 是否可以进入多 seed；
10. 是否可以进入 rainfall-scaling budget scenario；
11. 如果仍失败，下一步是否需要改变 action design，例如从 daily continuous action 改为 scheduled discrete action。

---

## 17. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_episode_profit_reward_debug_report.md
```

生成 PPT：

```text
docs/2026-06-06_episode_profit_reward_debug_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/episode_profit_reward_debug/reports/
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
git add prompts/006_05_redesign_episode_level_profit_reward.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/episode_profit_reward_debug/configs/
git add Leave_One_experiments/episode_profit_reward_debug/reward_versions/
git add Leave_One_experiments/episode_profit_reward_debug/evaluation/
git add Leave_One_experiments/episode_profit_reward_debug/figures/
git add Leave_One_experiments/episode_profit_reward_debug/reports/
git add docs/2026-06-06_episode_profit_reward_debug_report.md
git add docs/2026-06-06_episode_profit_reward_debug_report.pptx
git commit -m "Redesign episode-level profit reward for action-safe PPO"
```

注意：

1. 不要默认 commit 大模型 `.zip`；
2. 不要默认 commit tensorboard 大日志；
3. 不要默认 commit 过大的 daily_outputs；
4. 不要强行 push。

---

## 19. 完成后请汇报

完成后请汇报：

1. terminal reward 接口是否可实现；
2. P0-P6 哪些被测试；
3. 是否有 candidate 让 HLA 不再打满 300/450；
4. 哪些 candidate 产量损失 <= 15%；
5. 推荐 candidate 是哪个；
6. SYA/LCA 小扩展是否执行；
7. 是否可以进入多 seed；
8. 是否可以进入 rainfall-scaling budget scenario；
9. 如果仍失败，是否需要改 action design。
