# 006_17_all_year_direct_action_safe_ppo_simple_baseline

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_15_use_2020_2023_weather_and_all_available_years_for_calibration_validation_and_PPO.md
docs/2026-06-06_all_year_weather_calibration_validation_report.md
```

如果文件名或日期略有不同，请搜索关键词：

```text
all_year_weather_calibration_validation_report
all_year_weather_scenario_pool
weather_year_inventory
water_stress_year
irrigation_responsive_year
2020
2021
2022
2023
```

现在执行新的子任务：**基于 006_15 建立的 all-year weather scenario pool，直接训练一套简易版有约束 PPO 水氮管理策略。**

本任务不再继续 RF / imitation prior / expert replay / offline schedule search 路线。本任务目标是满足导师当前要求：**用 PPO 做出一套可解释、可展示的水氮优化策略，并用四套图证明 PPO 在合理时间进行了合理决策，最后得到合理作物生长结果。**

---

## 1. 本阶段核心目标

本阶段只做一件事：

```text
用所有可用年份气象数据构建训练/验证年份集合，
直接训练 action-safe PPO，
输出 PPO 水氮决策结果，
并生成四套图展示 PPO 策略合理性。
```

四套图是：

```text
1. 实际天气图：从种植前一个月开始的降水变化图；
2. 土壤水分/氮胁迫图：SWFAC / NSTRES 时间序列；
3. PPO 实际决策图：每日灌溉 / 施肥动作；
4. 作物生长图：TOPWT / GRNWT 时间序列。
```

本任务的最终目标不是证明 RF 好，也不是证明 expert schedule 好，而是形成一套：

```text
all-year direct action-safe PPO strategy
```

并用图和表说明：

```text
PPO 在科学时间进行了科学的灌溉/施肥决策，
缓解了水分/氮胁迫，
获得了合理的 TOPWT / GRNWT 结果。
```

---

## 2. 本阶段不要做的事情

本阶段明确不要做以下内容：

1. 不要做 RF / random forest imitation prior。
2. 不要训练 behavior cloning。
3. 不要做 expert replay。
4. 不要做 offline schedule search。
5. 不要做 expert dataset augmentation。
6. 不要做 constrained PPO fine-tuning from prior。
7. 不要做 replay consistency debug。
8. 不要做 episode-level profit reward。
9. 不要做 reward cost coefficient tuning。
10. 不要做低频决策 / 物候窗口 action design。
11. 不要做显式季节预算动作。
12. 不要做 scheduled event action。
13. 不要和 RF / expert policy 做对比。
14. 不要做 rainfall-scaling。
15. 不要训练无 action safety PPO。
16. 不要修改 `my_data/` 原始文件。
17. 不要覆盖 006_08 到 006_16 的结果。
18. 不要默认 commit 大模型、tensorboard 或大量 daily_outputs。

如果简易版 PPO 效果不好，先在报告中如实说明，不要在本任务中自动进入上面这些复杂路线。

---

## 3. 本阶段允许使用的约束

本阶段只允许使用简单、清楚、可解释的约束：

### 3.1 Action scale 约束

必须使用 action scaling / action safety，避免 PPO 输出不现实的水氮量。

建议范围：

```text
daily_irrigation_max = 40 mm/day
daily_n_max = 80 kg/ha/day
season_irrigation_cap = 160 mm
season_n_cap = 250 kg/ha
```

如果代码已有成熟 action safety wrapper，请复用；如果没有，请实现最小版本。

要求每日 CSV 中记录：

```text
raw_action_amir
raw_action_anfer
scaled_action_amir
scaled_action_anfer
safe_action_amir
safe_action_anfer
season_cumulative_irrigation
season_cumulative_n
action_safety_triggered
```

### 3.2 固定经济成本

允许使用固定经济成本进入 reward，但不要做成本系数调参。

固定成本建议：

```text
water_cost = 0.1
nitrogen_cost = 0.25
```

说明：

```text
这些是 normalized cost，不是真实人民币；
本阶段不调参，只作为简单经济约束，防止 PPO 无限制施水施肥。
```

### 3.3 简易 reward

本阶段不要做 episode-level profit reward，也不要做多套 reward 调参。

建议只实现一个固定 reward，例如：

```text
reward = yield_or_growth_reward - water_cost * irrigation - nitrogen_cost * nitrogen
```

可选写法：

```text
growth_reward = delta_grnwt + 0.1 * delta_topwt
reward = growth_reward - 0.1 * irrigation - 0.25 * nitrogen
```

如果 `grnwt` 在早期长期为 0，可以使用：

```text
growth_reward = 0.1 * delta_topwt + delta_grnwt
```

要求：

1. 只用这一套 reward；
2. 不要调多组成本系数；
3. 不要做 reward candidate grid；
4. 报告中必须写清楚 reward 公式和单位含义；
5. 如果 reward 信号不稳定，只报告问题，不要本任务中切换到复杂 reward。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_all_year_weather_calibration_validation_report.md

Leave_One_experiments/all_year_weather_calibration_validation/weather_inventory/weather_year_inventory.csv
Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/all_year_weather_scenario_pool.csv
Leave_One_experiments/all_year_weather_calibration_validation/stress_diagnostics/all_year_fixed_management_stress_summary.csv
Leave_One_experiments/all_year_weather_calibration_validation/cultivar_calibration_plan/weather_2020_2023_availability.csv
Leave_One_experiments/all_year_weather_calibration_validation/cultivar_calibration_plan/cultivar_calibration_validation_plan.md

weather_clean_qc/
Leave_One_experiments/wth_generated_qc/
my_data/
experiments/ppo_observed_years/
src/
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/
```

建议目录结构：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/
  configs/
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
docs/2026-06-06_all_year_direct_action_safe_ppo_report.md
docs/2026-06-06_all_year_direct_action_safe_ppo_report.pptx
```

---

## 6. 第一步：选择 PPO 训练/验证年份集合

基于：

```text
Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/all_year_weather_scenario_pool.csv
```

选择训练和验证年份。

输出：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/configs/ppo_train_eval_year_selection.csv
Leave_One_experiments/all_year_direct_action_safe_ppo/configs/ppo_train_eval_year_selection.md
```

字段至少包括：

```text
station_code
station_name
year
weather_file
scenario_type
has_water_stress
has_nitrogen_stress
irrigation_responsive
recommended_for_ppo_train
recommended_for_ppo_eval
selected_for_train
selected_for_eval
selection_reason
```

选择原则：

1. 优先包含 `water_stress_year`。
2. 优先包含 `irrigation_responsive_year`。
3. 必须包含 `nitrogen_stress_year`。
4. 必须包含 normal / wet / dry 的混合年份。
5. 2020–2023 年要保留为 calibration / validation 相关年份，不要丢掉。
6. 如果计算量有限，每个站点先选择 6–10 年：
   ```text
   train: 4–7 年
   eval: 2–3 年
   ```
7. 如果计算量允许，每个站点选择更多年份。
8. 不要只用 observed years。

建议初始方案：

```text
每站点：
  train_years:
    water_stress_years 中 2–3 年
    nitrogen_stress_years 中 1–2 年
    normal_year 中 1–2 年
    2020/2021 若可用则作为 calibration-related train candidates

  eval_years:
    irrigation_responsive_year 或 dry_year 中 1 年
    normal_year 中 1 年
    2022/2023 若可用则作为 validation-related eval candidates
```

如果某站点年份不足，请按实际情况调整并写明。

---

## 7. 第二步：实现 all-year direct action-safe PPO 配置

新增或更新配置：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/configs/config_all_year_direct_action_safe_ppo.yaml
```

配置至少包括：

```text
mode: water_nitrogen
algorithm: PPO
seed: 0
total_timesteps: 10000
action_safety_enabled: true
daily_irrigation_max: 40
daily_n_max: 80
season_irrigation_cap: 160
season_n_cap: 250
water_cost: 0.1
nitrogen_cost: 0.25
reward_type: simple_growth_minus_fixed_economic_cost
```

如果 10000 timesteps 计算量太大，先使用：

```text
total_timesteps: 5000
```

但报告中要说明。

要求：

1. PPO 直接从 gym-DSSAT 状态学习；
2. 不接 RF prior；
3. 不接 expert replay；
4. 不接 offline schedule action；
5. 保留 action scale 和 action safety；
6. reward 只用一套固定公式；
7. 每个站点可以单独训练一个 PPO；
8. 不要把所有站点混在一个环境里导致难以排错。

---

## 8. 第三步：实现训练脚本

新增或更新：

```text
src/train_all_year_direct_action_safe_ppo.py
```

要求：

1. 支持按站点训练；
2. 支持多个 train_years；
3. 每个 episode 使用一个 train_year；
4. train_year 可以按顺序循环或随机抽样；
5. 保留 seed=0；
6. 保存模型；
7. 保存 monitor log；
8. 保存 tensorboard log；
9. 保存训练配置快照；
10. 如果某个年份运行失败，要记录并跳过，不要让整个训练静默崩溃。

输出模型路径：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/models/{station_code}/ppo_direct_action_safe_seed0.zip
```

日志路径：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/logs/{station_code}/
Leave_One_experiments/all_year_direct_action_safe_ppo/tensorboard/{station_code}/
```

训练摘要：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/evaluation/training_run_summary.csv
```

字段至少包括：

```text
station_code
train_years
seed
total_timesteps
run_status
model_path
mean_train_reward
final_train_reward
notes
```

---

## 9. 第四步：实现评估脚本

新增或更新：

```text
src/evaluate_all_year_direct_action_safe_ppo.py
```

要求：

1. 对每个站点的 PPO 模型进行 train_years 和 eval_years 评估；
2. 每个站点-年份输出完整 daily CSV；
3. daily CSV 必须包含天气、状态、动作、生长变量；
4. 不要只输出 summary。

daily CSV 输出：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/{station_code}/{year}_ppo_daily.csv
```

daily CSV 字段至少包括：

```text
station_code
year
date
doy
dap
rain
srad
tmax
tmin
swfac
nstres
topwt
grnwt
xlai
reward
raw_action_amir
raw_action_anfer
scaled_action_amir
scaled_action_anfer
safe_action_amir
safe_action_anfer
season_cumulative_irrigation
season_cumulative_n
action_safety_triggered
```

summary 输出：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/evaluation/ppo_direct_eval_summary.csv
```

字段至少包括：

```text
station_code
station_name
year
split
scenario_type
run_status
episode_completed
final_grnwt
final_topwt
max_xlai
total_irrigation
total_n
profit_simple
mean_swfac
max_swfac
swfac_stress_days_gt_0p05
mean_nstres
max_nstres
nstres_days_gt_0p05
irrigation_event_count
n_event_count
first_irrigation_dap
first_n_dap
model_path
daily_csv_path
notes
```

profit_simple：

```text
profit_simple = 0.01 * final_grnwt - 0.1 * total_irrigation - 0.25 * total_n
```

---

## 10. 第五步：生成导师需要的四套结果图

对每个站点的代表性年份生成四套图。

代表性年份至少包括：

```text
1 个 water_stress_year
1 个 irrigation_responsive_year
1 个 nitrogen_stress_year
1 个 normal_year
1 个 2020–2023 validation year
```

如果某站点没有某类年份，报告说明。

新增或更新：

```text
src/plot_all_year_direct_ppo_four_panel_results.py
```

每个站点-年份生成一个四联图，也可以生成四张独立图。

图 1：实际天气降水变化图

```text
标题：Actual rainfall from one month before planting
内容：
  x = date 或 DAP
  y = daily rain
  叠加 cumulative rain 可选
范围：
  从 planting_date 前 30 天开始
  到 harvest / maturity / season end
```

图 2：实际土壤水分/氮胁迫指数变化图

```text
标题：SWFAC and NSTRES dynamics
内容：
  x = DAP
  y1 = swfac
  y2 = nstres
标出：
  swfac > 0.05 的时期
  nstres > 0.05 的时期
```

图 3：PPO 实际决策变化图

```text
标题：PPO irrigation and fertilization actions
内容：
  x = DAP
  y1 = safe_action_amir
  y2 = safe_action_anfer
  也可叠加 season cumulative irrigation / N
说明：
  灌溉和施肥动作必须使用 safe executed action，不要只画 raw action
```

图 4：TOPWT / GRNWT 变化图

```text
标题：Crop growth response under PPO strategy
内容：
  x = DAP
  y1 = topwt
  y2 = grnwt
```

图保存路径：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/figures/four_panel/{station_code}_{year}_four_panel.png
```

也要输出单独四类图：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/figures/weather/
Leave_One_experiments/all_year_direct_action_safe_ppo/figures/stress/
Leave_One_experiments/all_year_direct_action_safe_ppo/figures/actions/
Leave_One_experiments/all_year_direct_action_safe_ppo/figures/growth/
```

---

## 11. 第六步：生成 PPO 决策合理性诊断表

输出：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/evaluation/ppo_decision_reasonableness_diagnosis.csv
```

字段至少包括：

```text
station_code
year
scenario_type
total_irrigation
total_n
irrigation_event_count
n_event_count
swfac_stress_days_gt_0p05
nstres_days_gt_0p05
irrigation_during_or_before_swfac_stress
fertilization_during_or_before_nstres
first_irrigation_dap
first_n_dap
peak_swfac_dap
peak_nstres_dap
final_grnwt
final_topwt
profit_simple
decision_reasonableness_label
notes
```

建议判定：

```text
reasonable:
  不打满 season cap；
  存在水分胁迫年份中，灌溉发生在 swfac 高值之前或期间；
  存在氮胁迫年份中，施肥发生在 nstres 高值之前或期间；
  TOPWT/GRNWT 正常增长；
  final_grnwt 不异常低。

questionable:
  完全不施肥但 nstres 很高；
  完全不灌溉但 swfac 很高；
  大量水氮集中在不合理 DAP；
  final_grnwt 异常低。

cap_saturated:
  total_irrigation 接近 season_irrigation_cap；
  或 total_n 接近 season_n_cap。
```

---

## 12. 第七步：如果简易版 PPO 不理想，只报告，不自动扩展复杂路线

如果出现以下情况：

```text
PPO 打满 season cap；
PPO 完全不施肥导致 nstres 很高；
PPO 完全不灌溉导致 swfac 很高；
产量明显异常；
模型不收敛；
动作全部为 0；
```

请在报告中明确记录：

```text
simple direct action-safe PPO baseline 暂未达到可解释策略要求。
```

但本任务中不要自动进入：

```text
reward 成本项调参
episode-level profit reward
低频决策与物候窗口 action design
显式季节预算和事件动作
offline schedule search
constrained PPO fine-tuning
expert dataset augmentation
RF prior
```

只提出下一步可选建议即可。

---

## 13. 图表汇总

至少生成以下汇总图：

```text
ppo_total_irrigation_by_station_year.png
ppo_total_n_by_station_year.png
ppo_final_grnwt_by_station_year.png
ppo_profit_simple_by_station_year.png
ppo_swfac_nstres_stress_days_by_station_year.png
ppo_decision_reasonableness_summary.png
```

保存到：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/figures/summary/
```

---

## 14. 报告要求

报告必须说明：

1. 本任务为什么采用 direct action-safe PPO；
2. 训练/验证年份如何选择；
3. action scale 和 action safety 如何设置；
4. 简易 reward 公式是什么；
5. 每个站点 PPO 训练是否成功；
6. 每个站点-年份评估结果；
7. PPO 是否打满 cap；
8. PPO 是否在合理时间灌溉/施肥；
9. 四套图展示结果；
10. 是否能形成一套可向导师展示的 PPO 策略；
11. 如果不理想，下一步建议是什么，但不要在本任务自动执行复杂路线。

---

## 15. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_all_year_direct_action_safe_ppo_report.md
```

生成 PPT：

```text
docs/2026-06-06_all_year_direct_action_safe_ppo_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/all_year_direct_action_safe_ppo/reports/
```

PPT 重点放图，不要塞太多代码。建议结构：

```text
1. 研究目标：直接用 PPO 学水氮策略；
2. 年份选择：all-year scenario pool；
3. action scale / economic cost 约束；
4. 训练设置；
5. 代表性站点-年份四联图；
6. PPO 决策合理性；
7. 总结：是否形成可展示 PPO 策略；
8. 下一步建议。
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
git add prompts/006_17_all_year_direct_action_safe_ppo_simple_baseline.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/all_year_direct_action_safe_ppo/configs/
git add Leave_One_experiments/all_year_direct_action_safe_ppo/evaluation/
git add Leave_One_experiments/all_year_direct_action_safe_ppo/figures/
git add Leave_One_experiments/all_year_direct_action_safe_ppo/reports/
git add docs/2026-06-06_all_year_direct_action_safe_ppo_report.md
git add docs/2026-06-06_all_year_direct_action_safe_ppo_report.pptx
git commit -m "Train all-year direct action-safe PPO baseline"
```

注意：

1. 不要默认 commit 大量 daily_outputs；
2. 不要默认 commit 大模型 `.zip`；
3. 不要默认 commit tensorboard 大日志；
4. 不要强行 push。

---

## 17. 完成后请汇报

完成后请汇报：

1. 选择了哪些站点和年份训练 PPO；
2. 每个站点 PPO 是否训练成功；
3. 每个站点-年份总灌溉和总施氮是多少；
4. PPO 是否打满 cap；
5. PPO 是否在 swfac/nstres 胁迫附近做出决策；
6. TOPWT/GRNWT 是否正常增长；
7. 四套图是否全部生成；
8. 哪些代表性结果最适合给导师展示；
9. 简易版 PPO 是否已经足够；
10. 如果不够，下一步建议是什么。
