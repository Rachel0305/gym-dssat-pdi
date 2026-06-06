# 006_04_tune_reward_cost_coefficients_until_unsaturated

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_02_season_cap_sensitivity_analysis.md
prompts/006_03_reward_cost_revision_and_debug.md
docs/2026-06-06_season_cap_sensitivity_analysis_report.md
docs/2026-06-06_reward_cost_revision_and_debug_report.md
```

现在执行新的子任务：继续调试 reward 成本系数，直到至少在 HLA pilot 中出现“水氮不再打满 300/450 cap、且产量损失可接受”的 reward candidate。

本阶段不要做多 seed，不要进入 rainfall-scaling budget scenario，不要训练五站点，不要训练无 action safety 的 PPO。

---

## 1. 本阶段背景

上一阶段 `006_03_reward_cost_revision_and_debug` 测试了 current_reward_baseline、candidate_A1/A2/A3、candidate_B1/B2/B3、candidate_C1。测试设置为 HLA，train_year=2011，eval_years=2011/2007/2009，action safety cap=300 mm irrigation / 450 kg ha-1 N，timesteps=5000，seed=0。

结果显示：所有 candidates 的 HLA eval episodes 都完成，但所有 candidates 的 mean_irrigation=300 mm、mean_n=450 kg/ha、saturation ratio=1.0；没有任何 candidate 通过初筛，SYA/LCA extension 被跳过。

这说明上一轮成本系数仍然太弱，或者 reward scaling 不合理，PPO 仍然认为打满 300/450 cap 是最优行为。

---

## 2. 本阶段目标

1. 系统调大水氮成本系数。
2. 检查是否存在让 PPO 主动减少水氮投入的成本区间。
3. 找出至少 1 个 HLA candidate 满足：mean_irrigation_saturation_ratio < 0.95，mean_n_saturation_ratio < 0.95，yield_loss_vs_baseline <= 10% 或 <= 15%，所有 evaluation run_status=ok。
4. 如果没有 candidate 同时满足产量和投入条件，至少找出 trade-off frontier。
5. 如果成本一调高产量就崩，说明 reward 结构需要重写，而不是只调系数。
6. 只有 HLA 找到可用 candidate 后，才扩展到 SYA/LCA 小测试。

---

## 3. 禁止事项

1. 不要做多 seed。
2. 不要进入 rainfall-scaling budget scenario。
3. 不要训练 FQA/YCA。
4. 不要训练五站点。
5. 不要训练无 action safety 的 PPO。
6. 不要修改 `my_data/` 原始文件。
7. 不要覆盖 site-packages 中的原始 reward 文件。
8. 不要覆盖 `006_03` 的结果。
9. 不要把本阶段 reward 直接写成最终论文方法。
10. 不要只看 mean_reward，必须同时看 yield、water、N、saturation ratio、swfac、nstres。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_reward_cost_revision_and_debug_report.md
Leave_One_experiments/reward_cost_debug/evaluation/reward_candidate_evaluation_summary.csv
Leave_One_experiments/reward_cost_debug/evaluation/reward_candidate_comparison.csv
Leave_One_experiments/reward_cost_debug/reward_versions/current_reward_review.md
Leave_One_experiments/reward_cost_debug/reward_versions/reward_candidate_design.md
src/reward_candidates.py
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
Leave_One_experiments/reward_cost_tuning/
```

建议目录结构：

```text
Leave_One_experiments/reward_cost_tuning/
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
docs/2026-06-06_reward_cost_coefficient_tuning_report.md
docs/2026-06-06_reward_cost_coefficient_tuning_report.pptx
```

---

## 6. 第一步：复盘 006_03 为什么失败

请生成：

```text
Leave_One_experiments/reward_cost_tuning/evaluation/previous_candidate_failure_review.md
```

内容至少包括：上一轮每个 candidate 的 mean_irrigation、mean_n、yield、reward；哪些成本项被加了；为什么仍然打满 300/450；是否可能是成本项尺度远小于 crop reward；是否可能是 terminal yield reward 设计不足；是否可能是 PPO 训练步数太短导致策略未充分响应；本轮如何改进。

---

## 7. 第二步：新增更强成本候选

请不要覆盖上一轮 reward candidate。新增 reward 版本到：

```text
Leave_One_experiments/reward_cost_tuning/reward_versions/reward_candidates_v2.py
src/reward_candidates_v2.py
Leave_One_experiments/reward_cost_tuning/reward_versions/reward_candidate_v2_design.md
```

---

## 8. candidate_D：强线性成本扫描

在 candidate_A 的基础上显著增大成本系数。

```text
D1: irrigation_cost_coef = 0.5, nitrogen_cost_coef = 0.25
D2: irrigation_cost_coef = 1.0, nitrogen_cost_coef = 0.5
D3: irrigation_cost_coef = 2.0, nitrogen_cost_coef = 1.0
D4: irrigation_cost_coef = 5.0, nitrogen_cost_coef = 2.5
```

形式：

```text
reward = crop_growth_reward
         - irrigation_cost_coef * daily_irrigation
         - nitrogen_cost_coef * daily_n
```

目的：判断简单线性成本是否能让 PPO 对水氮投入敏感。

---

## 9. candidate_E：强二次累计成本

使用累计投入的二次惩罚，让越接近高投入区，边际成本越高。

```text
E1: cumulative_irrigation_coef = 0.0005, cumulative_n_coef = 0.0002
E2: cumulative_irrigation_coef = 0.001, cumulative_n_coef = 0.0005
E3: cumulative_irrigation_coef = 0.005, cumulative_n_coef = 0.002
E4: cumulative_irrigation_coef = 0.01, cumulative_n_coef = 0.005
```

形式：

```text
reward = crop_growth_reward
         - daily_linear_cost
         - cumulative_irrigation_coef * cumulative_irrigation^2
         - cumulative_n_coef * cumulative_n^2
```

目的：避免 PPO 在整个 season 中持续堆水氮。

---

## 10. candidate_F：目标区间惩罚

不再只惩罚超过 200/300，而是设置推荐投入区间，越偏离区间越扣分。

建议先用：

```text
target_irrigation_range = 100-220 mm
target_n_range = 150-320 kg/ha
```

形式：

```text
if cumulative_irrigation > upper:
    penalty = coef * (cumulative_irrigation - upper)^2

if cumulative_n > upper:
    penalty = coef * (cumulative_n - upper)^2
```

候选：

```text
F1: weak target penalty
F2: medium target penalty
F3: strong target penalty
```

---

## 11. candidate_G：经济学近似 reward

设计一个可解释的经济学近似版本，但先使用归一化或相对单位，避免真实价格不确定导致尺度爆炸。

形式：

```text
daily_reward = - water_cost * daily_irrigation - n_cost * daily_n
terminal_reward = grain_value * final_grain_yield
total_episode_objective = terminal_reward + sum(daily_reward)
```

候选：

```text
G1: grain_value = 0.01, water_cost = 0.2, n_cost = 0.1
G2: grain_value = 0.01, water_cost = 0.5, n_cost = 0.25
G3: grain_value = 0.005, water_cost = 0.5, n_cost = 0.25
G4: grain_value = 0.005, water_cost = 1.0, n_cost = 0.5
```

报告中说明单位目前是 normalized score，不是真实人民币；后续可替换为真实水价、氮肥价格和玉米价格。

---

## 12. 第三步：分批运行 HLA pilot

测试设置仍为：

```text
station = HLA
train_year = 2011
eval_years = 2011, 2007, 2009
cap = 300 mm / 450 kg ha-1 N
seed = 0
timesteps = 5000
action_safety_enabled = true
```

按批次执行：

```text
batch 1: D1, D2, D3, D4
batch 2: E1, E2, E3, E4
batch 3: F1, F2, F3
batch 4: G1, G2, G3, G4
```

如果某一批已经找到多个可用 candidate，可以停止后续批次，但必须说明为什么停止。若 D 系列没有任何 candidate 降低 saturation ratio 到 <0.95，则继续 batch 2；依此类推。

---

## 13. 每个 candidate 训练前 smoke check

每个 candidate 训练前，必须运行：

```text
HLA 2011 null_zero
HLA 2011 fixed_low_input
```

保存到：

```text
Leave_One_experiments/reward_cost_tuning/smoke_checks/
Leave_One_experiments/reward_cost_tuning/smoke_checks/pretrain_smoke_check_summary.csv
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
Leave_One_experiments/reward_cost_tuning/daily_outputs/HLA/
```

输出 summary 到：

```text
Leave_One_experiments/reward_cost_tuning/evaluation/reward_cost_tuning_evaluation_summary.csv
```

字段至少包括：

```text
station
reward_version
reward_family
batch
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
daily_csv_path
figure_dir
notes
```

---

## 15. 判断标准

每个 candidate 训练后计算：

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
Leave_One_experiments/reward_cost_tuning/evaluation/reward_cost_tuning_candidate_comparison.csv
```

初筛标准：

```text
mean_irrigation_saturation_ratio < 0.95
mean_n_saturation_ratio < 0.95
yield_loss_vs_baseline <= 10%
all episodes ok
```

宽松备选标准：

```text
mean_irrigation_saturation_ratio < 0.95
mean_n_saturation_ratio < 0.95
yield_loss_vs_baseline <= 15%
all episodes ok
```

如果只有单项不打满，例如水下降但氮仍打满，也记录为 partial_pass。

---

## 16. 图表输出

每个 candidate + eval_year 至少生成：

```text
daily_actions_raw_vs_safe.png
cumulative_water_nitrogen.png
crop_growth_timeseries.png
dap_swfac_irrigation_reward.png
dap_nstres_fertilization_reward.png
```

汇总图至少包括：

```text
reward_tuning_yield_vs_input.png
reward_tuning_saturation_ratio.png
reward_tuning_yield_loss_vs_input_reduction.png
reward_tuning_reward_vs_yield.png
reward_tuning_swfac_nstres.png
```

保存到：

```text
Leave_One_experiments/reward_cost_tuning/figures/
```

---

## 17. 如果找到有效 candidate，做 SYA/LCA 小扩展

如果 HLA 找到至少一个 candidate 满足初筛或宽松备选标准，选择最优 candidate 扩展到：

```text
SYA train_year = 2012
eval_years = 2012, 2014, 2015

LCA train_year = 2010
eval_years = 2010, 2011, 2008, 2009
```

输出：

```text
Leave_One_experiments/reward_cost_tuning/evaluation/cross_site_reward_cost_tuning_test.csv
```

如果 HLA 没有任何有效 candidate，不要扩展 SYA/LCA。

---

## 18. 报告要求

报告必须明确说明：

1. 上一轮 006_03 为什么失败；
2. 本轮新增了哪些更强成本候选；
3. 哪一批 candidate 开始使 PPO 不再打满 cap；
4. 是否出现产量崩塌；
5. 是否存在可接受 trade-off；
6. 推荐 candidate 是哪个；
7. 推荐 candidate 是否通过 SYA/LCA 小测试；
8. 如果仍无 candidate 通过，下一步应重构 reward，而不是继续简单调系数；
9. 是否可以进入多 seed；
10. 是否可以进入 rainfall-scaling budget scenario。

---

## 19. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_reward_cost_coefficient_tuning_report.md
```

生成 PPT：

```text
docs/2026-06-06_reward_cost_coefficient_tuning_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/reward_cost_tuning/reports/
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
git add prompts/006_04_tune_reward_cost_coefficients_until_unsaturated.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/reward_cost_tuning/configs/
git add Leave_One_experiments/reward_cost_tuning/reward_versions/
git add Leave_One_experiments/reward_cost_tuning/evaluation/
git add Leave_One_experiments/reward_cost_tuning/figures/
git add Leave_One_experiments/reward_cost_tuning/reports/
git add docs/2026-06-06_reward_cost_coefficient_tuning_report.md
git add docs/2026-06-06_reward_cost_coefficient_tuning_report.pptx
git commit -m "Tune reward cost coefficients for action-safe PPO"
```

注意：不要默认 commit 大模型 `.zip`；不要默认 commit tensorboard 大日志；不要默认 commit 过大的 daily_outputs；不要强行 push。

---

## 21. 完成后请汇报

完成后请汇报：

1. 哪些候选被测试；
2. 是否有 candidate 让 HLA 不再打满 300/450；
3. 哪些 candidate 产量损失 <= 10% 或 <= 15%；
4. 推荐 candidate 是哪个；
5. SYA/LCA 小扩展是否执行；
6. SYA/LCA 是否也不再打满 cap；
7. 是否可以进入多 seed；
8. 是否可以进入 rainfall-scaling budget scenario；
9. 如果仍失败，下一步是否需要重构 reward 结构。