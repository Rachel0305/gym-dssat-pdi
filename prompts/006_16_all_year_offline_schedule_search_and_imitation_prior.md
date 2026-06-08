# 006_16_all_year_offline_schedule_search_and_imitation_prior

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_15_use_2020_2023_weather_and_all_available_years_for_calibration_validation_and_PPO.md
docs/2026-06-06_all_year_weather_calibration_validation_report.md
```

现在执行新的子任务：基于 006_15 建立的 all-year weather scenario pool，重新做全年份 offline schedule search，生成覆盖水分胁迫年、灌溉响应年、氮胁迫年和正常年的 expert schedules / expert trajectories，并重新训练 all-year imitation prior。

本阶段不要直接训练 PPO，不要做 rainfall-scaling，不要训练无 action safety PPO。先把 all-year expert prior / imitation prior 建好。

---

## 1. 本阶段背景

`006_15_use_2020_2023_weather_and_all_available_years_for_calibration_validation_and_PPO` 已经完成。

关键结果：

```text
Weather inventory:
  119 station-years with QC weather/WTH data

2020–2023:
  all five stations available

All-year fixed management diagnostics:
  water-stress years = 20
  irrigation-responsive years = 6

Observed-year-only:
  water-stress years = 3
  irrigation-candidate years = 1
```

这说明：

```text
使用所有可用年份后，水分胁迫样本明显增加；
现在可以重新做 all-year offline schedule search；
目标是构建更丰富的 expert trajectories，再训练 all-year imitation prior。
```

报告推荐下一阶段：

```text
006_16_all_year_offline_schedule_search_and_imitation_prior
```

---

## 2. 本阶段一句话目标

本阶段要完成：

```text
基于 all-year scenario pool，
对 water-stress / irrigation-responsive / nitrogen-stress / normal years 做 offline schedule search，
构建 all-year expert library 和 imitation dataset，
重新训练 BC/RF/two-stage imitation prior，
并判断是否可以进入 all-year constrained PPO。
```

---

## 3. 本阶段不要做的事情

1. 不要直接训练 PPO。
2. 不要训练 unrestricted PPO。
3. 不要训练无 action safety PPO。
4. 不要进入 rainfall-scaling。
5. 不要覆盖 006_08 到 006_15 的结果。
6. 不要修改 `my_data/` 原始文件。
7. 不要只用 observed years。
8. 不要只选 profit 最高 schedules。
9. 不要只保留 irrigation=0 schedules。
10. 不要只看 yield，必须同时看 profit、water、N、swfac、nstres。
11. 不要把 2020–2023 weather 直接当作最终 RL 结果；它们目前主要用于 calibration/validation 规划。
12. 不要跳过 all-year expert trajectory 质量检查。

---

## 4. 输入文件

优先读取：

```text
docs/2026-06-06_all_year_weather_calibration_validation_report.md

Leave_One_experiments/all_year_weather_calibration_validation/weather_inventory/weather_year_inventory.csv
Leave_One_experiments/all_year_weather_calibration_validation/cultivar_calibration_plan/weather_2020_2023_availability.csv
Leave_One_experiments/all_year_weather_calibration_validation/cultivar_calibration_plan/cultivar_calibration_validation_plan.md
Leave_One_experiments/all_year_weather_calibration_validation/stress_diagnostics/all_year_fixed_management_stress_summary.csv
Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/all_year_weather_scenario_pool.csv
Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/scenario_pool_summary.md
Leave_One_experiments/all_year_weather_calibration_validation/evaluation/new_experiment_route_after_group_meeting.md

src/offline_schedule_policy.py
src/run_offline_schedule_search.py
src/run_all_year_fixed_management_stress_diagnostic.py
src/train_imitation_policy.py
src/imitation_policy_models.py
src/evaluate_imitation_policy.py
src/replay_imitation_prior.py
src/ppo_action_safety.py
```

如果路径不同，请搜索文件名，不要猜。

---

## 5. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/all_year_offline_schedule_search/
```

建议目录结构：

```text
Leave_One_experiments/all_year_offline_schedule_search/
  configs/
  candidate_schedules/
  daily_outputs/
  evaluation/
  expert_policy/
  imitation_dataset/
  models/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-06_all_year_offline_schedule_search_and_imitation_prior_report.md
docs/2026-06-06_all_year_offline_schedule_search_and_imitation_prior_report.pptx
```

---

## 6. 第一步：选择 all-year search 年份集合

基于 `all_year_weather_scenario_pool.csv`，先建立 search year list。

输出：

```text
Leave_One_experiments/all_year_offline_schedule_search/configs/all_year_search_year_selection.csv
Leave_One_experiments/all_year_offline_schedule_search/configs/all_year_search_year_selection.md
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
selected_for_search
selected_for_train_pool
selected_for_eval_pool
selection_reason
```

选择原则：

1. 必须优先包含 `water_stress_year`、`irrigation_responsive_year`、`nitrogen_stress_year`、`dry_year`、`normal_year`。
2. all-year 报告中明确出现的 irrigation-responsive examples 包括但不限于：
   ```text
   FQA 2007
   FQA 2008
   FQA 2016
   HLA 2004
   SYA 2017
   YCA 2004
   ```
   请以实际 `all_year_weather_scenario_pool.csv` 为准。
3. 2020–2023 用于 calibration/validation 规划，不能丢掉：
   ```text
   每个站点尽量保留 2020–2021 作为 calibration candidates；
   每个站点尽量保留 2022–2023 作为 validation candidates；
   如果某年同时是 water-stress / irrigation-responsive / nitrogen-stress，需要在 selection_reason 中说明。
   ```
4. 如果全部 119 station-years 计算量太大，先做分层抽样：
   ```text
   water_stress_years: all
   irrigation_responsive_years: all
   dry_years: up to 3 per station
   normal_years: up to 4 per station
   wet_years: up to 3 per station
   2020–2023: all available
   ```
5. 目标规模建议：
   ```text
   每站点 10–16 个年份；
   全体 50–80 个 station-years；
   如果计算资源允许，可以全部 119 station-years。
   ```

---

## 7. 第二步：定义 all-year offline schedule search 空间

本阶段不训练 PPO，只做 deterministic schedule search。

### 7.1 氮肥事件

```text
N events:
  DAP 1
  DAP 30
  DAP 60

N amount candidates:
  [0, 50, 75, 100, 150]

total_n <= 250 kg/ha
```

### 7.2 灌溉事件

为 water-stress / irrigation-responsive years 使用更细的灌溉事件：

```text
I events:
  DAP 20
  DAP 35
  DAP 50
  DAP 65
  DAP 80
  DAP 95

I amount candidates:
  [0, 20, 40, 60]

total_irrigation <= 160 mm
```

### 7.3 非水分胁迫年份的简化搜索

对于 low_response_year / wet_year，如果 full grid 过大，可以用简化搜索：

```text
I events:
  DAP 35
  DAP 65

I amount:
  [0, 30, 60]
```

但仍必须保留 N 搜索。

### 7.4 分阶段搜索策略

如果组合过多，使用三阶段：

```text
Stage A:
  固定几个 N 总量或 N 分配，搜索 irrigation timing/amount

Stage B:
  固定 top irrigation schedules，搜索 N 分配

Stage C:
  围绕 top Pareto schedules 局部 refine
```

---

## 8. 第三步：运行 all-year deterministic offline schedule search

新增或更新：

```text
src/run_all_year_offline_schedule_search.py
```

要求：

1. 不训练 PPO；
2. 使用 deterministic schedule policy；
3. 对 selected station-years 运行 schedule search；
4. 所有 action 必须通过 action safety；
5. 保存 daily outputs；
6. 保存 schedule-level summary；
7. 记录 swfac/nstres；
8. 记录 profit_default 和 profit_low_water_cost；
9. 对 run stalled / failed 的站点年份必须记录原因，不要静默跳过。

输出：

```text
Leave_One_experiments/all_year_offline_schedule_search/evaluation/all_year_schedule_search_summary.csv
```

字段至少包括：

```text
station_code
station_name
year
weather_file
scenario_type
schedule_id
run_status
episode_completed
final_grnwt
final_topwt
final_xlai
total_irrigation
total_n
profit_default
profit_low_water_cost
mean_swfac
max_swfac
swfac_stress_days_gt_0p05
mean_nstres
max_nstres
nstres_days_gt_0p05
yield_gain_vs_no_irrigation_same_n
profit_gain_vs_no_irrigation_same_n_default
profit_gain_vs_no_irrigation_same_n_low_water_cost
yield_per_100mm_irrigation
n_productivity
daily_csv_path
notes
```

---

## 9. 第四步：选择 all-year expert schedules

不要只选 profit 最高。每个站点-年份至少尝试保留以下类型：

```text
top_profit_default
top_profit_low_water_cost
top_yield
low_input_within_5pct_yield_loss
low_input_within_10pct_yield_loss
irrigation_responsive
water_stress_relief
nitrogen_efficient
pareto_balanced
```

筛选标准：

```text
run_status ok
episode_completed true
total_n <= 250
total_irrigation <= 160
not dominated on yield-input-profit Pareto frontier
```

irrigation_responsive schedule 标准：

```text
total_irrigation > 0
yield_gain_vs_no_irrigation_same_n >= 200 kg/ha
或 yield_gain_vs_no_irrigation_same_n >= 3%
profit_low_water_cost 不明显低于 no-irrigation same-N
swfac 有改善或 yield 有改善
```

输出：

```text
Leave_One_experiments/all_year_offline_schedule_search/expert_policy/all_year_expert_schedule_library.csv
Leave_One_experiments/all_year_offline_schedule_search/evaluation/all_year_expert_library_summary.csv
```

---

## 10. 第五步：生成 all-year imitation dataset

从 selected expert schedules 的 daily outputs 构建：

```text
Leave_One_experiments/all_year_offline_schedule_search/imitation_dataset/imitation_dataset_all_year.csv
```

字段至少包括：

```text
station_code
station_name
year
weather_file
scenario_type
schedule_id
expert_type
date
doy
dap
state_variables
swfac
nstres
topwt
grnwt
xlai
expert_action_irrigation
expert_action_n
total_schedule_irrigation
total_schedule_n
profit_default
profit_low_water_cost
source_daily_csv
```

必须检查动作分布：

```text
Leave_One_experiments/all_year_offline_schedule_search/evaluation/all_year_imitation_dataset_action_distribution.csv
Leave_One_experiments/all_year_offline_schedule_search/evaluation/all_year_imitation_dataset_check.md
```

必须比较：

```text
00609_original
00612_augmented
00616_all_year
```

重点回答：

```text
all-year dataset 是否显著增加 nonzero irrigation rows；
是否覆盖多个站点、多个年份、多个 DAP 阶段；
是否包含 water-stress / irrigation-responsive scenarios。
```

---

## 11. 第六步：训练 all-year imitation prior

重新训练：

```text
BC_random_forest_regressor_all_year
BC_two_stage_classifier_regressor_all_year
BC_mlp_regressor_all_year
```

可选：

```text
BC_constant_schedule_baseline_all_year
```

新增或更新训练配置：

```text
Leave_One_experiments/all_year_offline_schedule_search/configs/train_all_year_imitation_prior.yaml
```

输出：

```text
Leave_One_experiments/all_year_offline_schedule_search/models/
Leave_One_experiments/all_year_offline_schedule_search/evaluation/all_year_imitation_supervised_metrics.csv
```

训练/验证 split 要求：

1. 不要随机泄漏同一 schedule 的相邻日；
2. 优先按 year / station-year / schedule 分组 split；
3. train pool 和 eval pool 要参考 `all_year_weather_scenario_pool.csv`；
4. 单独报告 water-stress / irrigation-responsive years 上的动作识别能力；
5. 重点关注 irrigation event recall，不要只看 overall accuracy。

指标至少包括：

```text
irrigation_mae
irrigation_rmse
irrigation_event_precision
irrigation_event_recall
irrigation_event_f1
nitrogen_mae
nitrogen_rmse
nitrogen_event_precision
nitrogen_event_recall
nitrogen_event_f1
nonzero_action_accuracy
zero_action_accuracy
water_stress_irrigation_event_recall
irrigation_responsive_event_recall
```

---

## 12. 第七步：在 DSSAT/gym-DSSAT 中评估 all-year imitation prior

对训练出的 policy 做 deterministic evaluation。

至少评估：

```text
BC_random_forest_regressor_all_year
BC_two_stage_classifier_regressor_all_year
BC_random_forest_regressor_augmented
best_expert_schedule_replay
old_ppo_cap_saturated_baseline
```

评估年份：

```text
recommended_for_ppo_eval=True 的 station-years
以及部分 recommended_for_ppo_train=True 的 station-years
```

输出：

```text
Leave_One_experiments/all_year_offline_schedule_search/evaluation/all_year_imitation_policy_dssat_summary.csv
```

字段至少包括：

```text
station_code
station_name
year
scenario_type
policy_name
run_status
episode_completed
final_grnwt
total_irrigation
total_n
profit_default
profit_low_water_cost
mean_swfac
max_swfac
swfac_stress_days_gt_0p05
mean_nstres
max_nstres
nstres_days_gt_0p05
yield_loss_vs_best_expert
profit_gap_vs_best_expert
input_reduction_vs_old_ppo
irrigation_event_count
n_event_count
daily_csv_path
notes
```

---

## 13. 第八步：选择推荐 all-year prior

输出：

```text
Leave_One_experiments/all_year_offline_schedule_search/evaluation/recommended_all_year_prior_policy.csv
```

推荐标准：

```text
run_status ok
不退化到 300/450
mean_n <= 250
mean_irrigation <= 160
yield_loss_vs_best_expert <= 15%
profit_low_water_cost 不明显差于 best expert
water_stress years 中能产生合理非零灌溉
irrigation_event_recall 明显高于 00612 augmented prior
行为可解释
```

如果没有 learned policy 通过，则推荐：

```text
继续改进 all-year expert library 或使用 deterministic expert schedule baseline；
不要进入 PPO。
```

如果有 learned policy 通过，则推荐下一阶段：

```text
006_17_all_year_constrained_ppo_with_all_year_prior
```

---

## 14. 图表输出

至少生成：

```text
all_year_schedule_search_pareto_by_station.png
all_year_expert_library_scenario_coverage.png
all_year_imitation_action_distribution.png
irrigation_event_dap_distribution.png
all_year_bc_supervised_metrics.png
all_year_policy_yield_vs_input.png
all_year_policy_profit_comparison.png
water_stress_year_policy_comparison.png
irrigation_responsive_year_policy_comparison.png
```

保存到：

```text
Leave_One_experiments/all_year_offline_schedule_search/figures/
```

---

## 15. 报告要求

报告必须说明：

1. 为什么 `006_15` 后可以进入 all-year offline schedule search；
2. 选择了哪些 station-years；
3. 是否包含 2020–2023 calibration/validation 年份；
4. schedule search 空间；
5. expert library 覆盖了哪些 scenario types；
6. 是否显著增加了 irrigation-positive schedules 和 nonzero irrigation rows；
7. all-year imitation dataset 与 00609/00612 的比较；
8. all-year BC/RF/two-stage 训练结果；
9. DSSAT/gym-DSSAT 实际评估结果；
10. all-year prior 是否能在 water-stress years 中产生合理灌溉；
11. 是否推荐进入 all-year constrained PPO；
12. 如果不推荐，下一步是什么。

---

## 16. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_all_year_offline_schedule_search_and_imitation_prior_report.md
```

生成 PPT：

```text
docs/2026-06-06_all_year_offline_schedule_search_and_imitation_prior_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/all_year_offline_schedule_search/reports/
```

---

## 17. GitHub 备份

完成后先运行：

```bash
git status
```

请告诉我建议提交哪些文件。

如果没有明显问题，请执行：

```bash
git add prompts/006_16_all_year_offline_schedule_search_and_imitation_prior.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/all_year_offline_schedule_search/configs/
git add Leave_One_experiments/all_year_offline_schedule_search/evaluation/
git add Leave_One_experiments/all_year_offline_schedule_search/expert_policy/
git add Leave_One_experiments/all_year_offline_schedule_search/imitation_dataset/
git add Leave_One_experiments/all_year_offline_schedule_search/figures/
git add Leave_One_experiments/all_year_offline_schedule_search/reports/
git add docs/2026-06-06_all_year_offline_schedule_search_and_imitation_prior_report.md
git add docs/2026-06-06_all_year_offline_schedule_search_and_imitation_prior_report.pptx
git commit -m "Run all-year offline schedule search and imitation prior"
```

注意：

1. 不要默认 commit 大量 daily_outputs；
2. 不要默认 commit 大模型；
3. 不要默认 commit tensorboard；
4. 不要强行 push。

---

## 18. 完成后请汇报

完成后请汇报：

1. 选了多少 station-years；
2. 包含多少 water-stress / irrigation-responsive years；
3. 选出多少 expert schedules；
4. nonzero irrigation rows 是否明显增加；
5. all-year imitation dataset 相比 00609/00612 是否更适合训练灌溉动作；
6. 哪个 all-year imitation policy 最好；
7. 它在 water-stress years 中是否会合理灌溉；
8. 是否推荐进入 all-year constrained PPO；
9. 如果不推荐，下一步是什么。
