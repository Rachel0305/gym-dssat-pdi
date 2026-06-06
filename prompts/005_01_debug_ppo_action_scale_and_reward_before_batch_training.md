# 003_07_debug_ppo_action_scale_and_reward_before_batch_training

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/003_06_generate_leave_one_ppo_training_framework.md
docs/2026-06-05_ppo_observed_year_training_framework.md
```

现在执行新的子任务：在批量 PPO 训练之前，排查 HLA 2007 debug PPO 的动作尺度、动作频率、reward 和评估输出是否合理。

本阶段不要批量训练 PPO。

---

## 1. 本阶段背景

上一阶段已经完成 PPO observed-year leave-one 框架生成，并且 HLA 2007 seed0 debug PPO 可以成功训练、评估和保存结果。

但是 HLA 2007 debug PPO 的结果存在严重异常：

```text
eval 2007:
total_irrigation = 3067.78
total_n_fertilizer = 14503.8

eval 2011:
total_irrigation = 3486.41
total_n_fertilizer = 16457.5

eval 2009:
total_irrigation = 3278.6
total_n_fertilizer = 15485.3
```

这些水氮投入量远高于固定策略 smoke test 中的合理范围，例如 fixed_high_input 只有：

```text
total_irrigation = 120 mm
total_n = 250 kg/ha
```

因此当前不能直接进入 HLA/SYA/LCA 批量 PPO 训练。

本阶段目标是：

1. 确认 PPO 动作是否每天都在施肥/灌溉；
2. 确认 action normalization / denormalization 是否正确；
3. 确认 `totir`、`tofer` 计算是否正确；
4. 检查 reward 是否在鼓励过量施肥/灌溉；
5. 检查 PPO action space 是否过宽；
6. 设计一个更安全的 PPO 动作约束和 reward 诊断框架；
7. 在 HLA 2007 上重新做一个小步数 debug PPO；
8. 只有当水氮投入进入合理范围后，才允许进入批量训练。

---

## 2. 本阶段不要做的事情

1. 不要批量训练所有站点。
2. 不要训练 HLA/SYA/LCA 全部模型。
3. 不要修改 `my_data/` 原始数据。
4. 不要删除或覆盖上一阶段 PPO debug 结果。
5. 不要直接把当前 HLA debug PPO 当作有效策略。
6. 不要只通过图表美化掩盖过量水氮问题。
7. 不要在没有诊断报告的情况下直接改 reward。
8. 不要把固定策略结果和 PPO 结果混为最终结论。
9. 不要上传大模型文件到 GitHub。

---

## 3. 输入文件

优先读取上一阶段结果：

```text
docs/2026-06-05_ppo_observed_year_training_framework.md
Leave_One_experiments/ppo_observed_years/evaluation/ppo_evaluation_summary.csv
Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2007_seed0_daily.csv
Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2011_seed0_daily.csv
Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2009_seed0_daily.csv
Leave_One_experiments/ppo_observed_years/models/HLA/HLA_train2007_seed0_debug.zip
experiments/ppo_observed_years/config_ppo_observed_years.yaml
src/ppo_train.py
src/ppo_evaluate.py
src/ppo_safe_rendering.py
```

同时读取固定策略 smoke test 结果作为合理性对照：

```text
Leave_One_experiments/smoke_tests/evaluation/smoke_test_summary.csv
Leave_One_experiments/smoke_tests_debug/evaluation/smoke_test_timeout_rerun_summary.csv
```

如果路径不同，请搜索文件名。

---

## 4. 本阶段输出目录

所有本阶段输出保存到：

```text
Leave_One_experiments/ppo_action_debug/
```

建议目录结构：

```text
Leave_One_experiments/ppo_action_debug/
  action_diagnostics/
  reward_diagnostics/
  config_versions/
  models/
  daily_outputs/
  evaluation/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-05_ppo_action_scale_and_reward_debug_report.md
docs/2026-06-05_ppo_action_scale_and_reward_debug_report.pptx
```

如果日期不方便自动获取，可以使用当前系统日期。

---

## 5. 第一步：诊断当前 PPO daily action

请读取上一阶段 HLA 2007 debug PPO 的三个 daily CSV：

```text
HLA_train2007_eval2007_seed0_daily.csv
HLA_train2007_eval2011_seed0_daily.csv
HLA_train2007_eval2009_seed0_daily.csv
```

对每个 eval_year 计算：

```text
days
nonzero_irrigation_days
nonzero_fertilization_days
max_daily_irrigation
max_daily_fertilization
mean_daily_irrigation
mean_daily_fertilization
sum_irrigation
sum_fertilization
median_normalized_action_amir
median_normalized_action_anfer
max_normalized_action_amir
max_normalized_action_anfer
min_normalized_action_amir
min_normalized_action_anfer
```

输出：

```text
Leave_One_experiments/ppo_action_debug/action_diagnostics/current_ppo_action_summary.csv
```

同时生成逐日动作图：

```text
Leave_One_experiments/ppo_action_debug/figures/current_ppo_daily_actions_eval2007.png
Leave_One_experiments/ppo_action_debug/figures/current_ppo_daily_actions_eval2011.png
Leave_One_experiments/ppo_action_debug/figures/current_ppo_daily_actions_eval2009.png
```

请重点判断：

1. PPO 是否几乎每天都在施肥；
2. PPO 是否几乎每天都在灌溉；
3. normalized action 是否长期接近 1；
4. real action 是否明显过大；
5. `totir` 和 `tofer` 是否只是每日动作累计；
6. 是否存在 action clipping 失败。

---

## 6. 第二步：检查 action_space 和动作转换

请输出 PPO 环境的 action space 检查表：

```text
Leave_One_experiments/ppo_action_debug/action_diagnostics/ppo_action_space_check.csv
```

字段至少包括：

```text
station
year
action_name
action_space_low
action_space_high
normalization_formula
denormalization_formula
example_normalized_minus1
example_normalized_0
example_normalized_plus1
notes
```

请确认：

1. `amir` 上限是多少；
2. `anfer` 上限是多少；
3. PPO 输出 `[-1, 1]` 如何转换成真实动作；
4. `normalized_action = 1` 对应多少 mm 灌溉和多少 kg/ha N；
5. 当前 PPO daily output 的 `real_action_*` 是否与转换公式一致。

---

## 7. 第三步：检查 reward 函数

请定位当前使用的 reward 函数文件和函数名。

不要直接修改 reward，先做诊断。

输出 reward 诊断报告：

```text
Leave_One_experiments/ppo_action_debug/reward_diagnostics/current_reward_function_review.md
```

报告至少包括：

1. 当前 reward 函数完整代码片段；
2. reward 用到了哪些变量；
3. 是否包含灌溉惩罚；
4. 是否包含施氮惩罚；
5. 惩罚强度是多少；
6. 是否使用了 `topwt` 增量；
7. 是否使用了 `grnwt` 增量；
8. 是否可能鼓励每天大动作；
9. 是否在早期 grnwt 为 0 时给出有效学习信号；
10. 当前 reward 与 fixed_high_input 和 PPO 过量投入现象是否一致。

如果 reward 文件来自 site-packages，请不要直接改原文件，先复制一份到项目内可控路径或写出修改建议。

---

## 8. 第四步：区分“动作记录异常”与“策略真的过量动作”

请对 HLA 2007 eval2007 做一个复核：

1. 读取 daily CSV；
2. 手动计算 `sum(real_action_amir)`；
3. 手动计算 `sum(real_action_anfer)`；
4. 与 summary 中 `total_irrigation`、`total_n_fertilizer` 对比；
5. 如果不一致，说明 summary 计算有问题；
6. 如果一致，说明 PPO 真的输出了过量动作。

输出：

```text
Leave_One_experiments/ppo_action_debug/action_diagnostics/total_input_recalculation_check.csv
```

---

## 9. 第五步：建立 PPO 安全动作约束方案

在不修改原始 my_data 的前提下，设计一个 PPO 训练时的 action safety wrapper 或 post-processing 逻辑。

建议先实现最小安全约束：

```text
daily_irrigation_max = 40 mm
daily_n_max = 80 kg/ha
season_irrigation_soft_limit = 200 mm
season_n_soft_limit = 300 kg/ha
min_days_between_irrigation = 7
min_days_between_fertilization = 10
fertilization_allowed_dap_range = 1-90
irrigation_allowed_dap_range = 1-120
```

注意：

1. 这些不是最终论文参数，只是 debug 安全约束；
2. 所有参数必须写入 YAML 配置；
3. 不要硬编码在函数内部；
4. 如果超过 seasonal soft limit，可以将后续动作裁剪为 0，或在 reward 中增加强惩罚；
5. 每次裁剪都必须记录到 daily CSV。

建议字段：

```text
raw_real_action_amir
raw_real_action_anfer
safe_real_action_amir
safe_real_action_anfer
action_clipped_amir
action_clipped_anfer
season_irrigation_so_far
season_n_so_far
safety_rule_triggered
```

请新增或更新：

```text
src/ppo_action_safety.py
```

并在 `ppo_train.py` 和 `ppo_evaluate.py` 中可选启用。

---

## 10. 第六步：设计 reward 修正候选，但不要覆盖原 reward

请生成一个候选 reward 文件或建议文档，不要直接覆盖原 reward。

输出：

```text
Leave_One_experiments/ppo_action_debug/reward_diagnostics/reward_revision_candidates.md
```

候选 reward 至少包括两类：

### 10.1 强成本惩罚型 reward

思路：

```text
reward = crop_growth_reward - irrigation_cost - nitrogen_cost - excessive_input_penalty
```

要求：

1. 使用每日生物量或产量增量；
2. 对灌溉和施氮设置明确成本；
3. 对超过 season limit 的水氮设置额外惩罚；
4. 避免每天大水大肥也能得到高 reward。

### 10.2 终局产量 + 过程成本型 reward

思路：

```text
daily_reward = - daily_input_cost
terminal_reward = final_yield_value - total_input_cost
```

要求：

1. 每天主要惩罚水氮投入；
2. episode 结束时给产量收益；
3. 适合后续经济学派解释；
4. 报告中说明优缺点。

本阶段可以生成候选，不一定马上替换。

---

## 11. 第七步：重新做 HLA 2007 安全动作 debug PPO

在完成动作诊断和安全约束后，请运行一个新的小步数 debug PPO。

命名为：

```text
HLA_train2007_seed0_action_safe_debug
```

配置：

```text
station = HLA
train_year = 2007
seed = 0
total_timesteps = 1000 或 5000
action_safety_enabled = true
reward = current_reward
```

注意：

1. 先只启用 action safety，不改 reward；
2. 目的是确认水氮投入是否能被限制到合理范围；
3. 如果 action safety 后水氮投入合理，再考虑 reward 修正；
4. 如果 action safety 后仍然异常，继续查 action conversion。

输出模型：

```text
Leave_One_experiments/ppo_action_debug/models/HLA/HLA_train2007_seed0_action_safe_debug.zip
```

评估年份：

```text
2007
2011
2009
```

保存 daily CSV 到：

```text
Leave_One_experiments/ppo_action_debug/daily_outputs/HLA/
```

保存 summary 到：

```text
Leave_One_experiments/ppo_action_debug/evaluation/action_safe_debug_evaluation_summary.csv
```

---

## 12. 合理性判断标准

重新 debug PPO 后，请判断是否进入合理范围。

建议暂定合理范围：

```text
total_irrigation <= 300 mm
total_n_fertilizer <= 400 kg/ha
```

更严格的范围后续再结合农学背景和站点管理记录确定。

如果新 PPO 仍然远超：

```text
total_irrigation > 300 mm
total_n_fertilizer > 400 kg/ha
```

则不要进入批量训练。

如果新 PPO 已经控制在合理范围，再建议进入下一阶段。

---

## 13. 图表输出

请至少生成：

```text
current_vs_action_safe_total_irrigation.png
current_vs_action_safe_total_n.png
current_vs_action_safe_final_grnwt.png
current_vs_action_safe_reward.png
current_vs_action_safe_daily_actions_eval2007.png
current_vs_action_safe_daily_actions_eval2011.png
current_vs_action_safe_daily_actions_eval2009.png
```

保存到：

```text
Leave_One_experiments/ppo_action_debug/figures/
```

---

## 14. 报告输出

请生成 Markdown 报告：

```text
docs/2026-06-05_ppo_action_scale_and_reward_debug_report.md
```

报告至少包括：

1. 为什么不能直接批量 PPO；
2. 当前 HLA debug PPO 的异常水氮投入；
3. 当前 PPO 每日动作诊断；
4. action_space 和动作转换检查；
5. total_irrigation / total_n_fertilizer 重算结果；
6. 当前 reward 函数诊断；
7. 是否发现动作记录或 summary 计算错误；
8. action safety 方案；
9. reward 修正候选；
10. HLA 2007 action-safe debug PPO 结果；
11. 新旧 PPO 对比；
12. 是否可以进入批量训练；
13. 如果不能，下一步还需要修正什么。

---

## 15. PPT 输出

请生成 PPT：

```text
docs/2026-06-05_ppo_action_scale_and_reward_debug_report.pptx
```

PPT 至少包括：

1. 问题背景；
2. 当前 PPO 异常水氮投入；
3. daily action 诊断图；
4. action_space 检查；
5. reward 函数诊断；
6. action safety 设计；
7. action-safe PPO debug 结果；
8. 新旧对比；
9. 下一步建议。

并复制一份到：

```text
Leave_One_experiments/ppo_action_debug/reports/
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
git add prompts/003_07_debug_ppo_action_scale_and_reward_before_batch_training.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/ppo_action_debug/
git add docs/2026-06-05_ppo_action_scale_and_reward_debug_report.md
git add docs/2026-06-05_ppo_action_scale_and_reward_debug_report.pptx
git commit -m "Debug PPO action scale and reward before batch training"
```

不要强行 push。

如果存在大模型文件或过大日志，请先报告，不要 commit 大文件。

---

## 17. 完成后请汇报

完成后请汇报：

1. 当前 PPO 是否每天大水大肥；
2. action_space 上限是多少；
3. normalized action 是否长期接近 1；
4. summary 中 total_irrigation / total_n_fertilizer 是否计算正确；
5. 当前 reward 是否可能鼓励过量投入；
6. action safety 是否成功限制水氮投入；
7. action-safe HLA 2007 debug PPO 是否成功；
8. 新 PPO 的 total_irrigation 和 total_n_fertilizer 是否进入合理范围；
9. 是否可以进入批量训练；
10. 如果不能，下一步需要改 action、reward 还是环境。

---

## 18. 最重要原则

1. 当前 PPO 能跑通，但结果不合理，不能直接扩大训练。
2. 先诊断动作和 reward，再训练更多模型。
3. 不要用 PPO 结果图好看掩盖水氮投入异常。
4. 所有动作裁剪和 safety rule 必须记录。
5. action safety 与 reward 修正要分开测试。
6. 本阶段只先启用 action safety，不直接替换 reward。
