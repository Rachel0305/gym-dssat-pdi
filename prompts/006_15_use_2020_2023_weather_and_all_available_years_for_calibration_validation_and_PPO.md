# 006_15_use_2020_2023_weather_and_all_available_years_for_calibration_validation_and_PPO

请阅读项目根目录下的：

```text
AGENTS.md
TASK_LEAVE_ONE_YEAR_STRATEGY.md
prompts/006_14_water_nitrogen_factorial_diagnosis_then_irrigation_responsive_search.md
docs/2026-06-06_water_nitrogen_factorial_diagnosis_report.md
```

现在执行新的子任务：根据组会后的新要求，把实验路线从“只用每个站点少数实测年份训练/验证”调整为“使用 2020–2023 年气象数据进行品种参数校正和验证，并筛选所有可用年份气象数据用于后续 PPO / expert prior 实验”。

本阶段的重点不是继续训练 PPO，而是先建立全年份气象数据清单、2020–2023 校正验证方案、以及所有年份的水分/氮胁迫 scenario pool。

---

## 1. 组会后的新要求

导师要求：

```text
1. 不再只用每个站点当前少数实测年份做实验；
2. 直接选用 2020–2023 年气象数据进行校正和验证；
3. 当年的气象主要用于校准品种参数；
4. 后续筛选并使用所有年份气象数据进行实验；
5. 这样可以扩大气象年份范围，提高出现土壤水分胁迫的概率；
6. 有助于 PPO 学习和训练。
```

因此，本阶段是路线调整任务，不是继续 006_14 的小修小补。

---

## 2. 本阶段一句话目标

本阶段要完成：

```text
整理 2020–2023 年气象数据用于品种参数校正/验证，
同时筛选所有可用年份气象数据，构建 all-year weather scenario pool，
诊断哪些年份存在水分胁迫/氮胁迫/灌溉响应，
为后续 PPO 训练建立更丰富的训练和验证年份集合。
```

---

## 3. 为什么要这样改

前面的 `006_14` 结论是：

```text
当前少数 observed years 中，产量提升主要由氮驱动；
大多数站点-年份没有明显 swfac 水分胁迫；
只有少数年份出现轻微灌溉响应；
因此 observed-year-only 不适合直接训练灌溉 PPO。
```

但 observed-year-only 年份太少，水分胁迫出现概率低。

所以现在应该扩大年份范围：

```text
少数 observed years
↓
2020–2023 calibration / validation weather years
↓
all available weather years scenario pool
↓
water-stress / nitrogen-stress / irrigation-responsive year screening
↓
后续 all-year offline schedule search / imitation prior / PPO training
```

---

## 4. 本阶段核心问题

请回答：

1. 当前项目里每个站点有哪些可用年份的 WTH / 气象数据？
2. 2020–2023 年每个站点的数据是否齐全？
3. 2020–2023 年是否适合用于品种参数校正和验证？
4. 所有可用年份中，哪些年份更容易出现水分胁迫？
5. 哪些年份更适合用于 PPO 训练？
6. 哪些年份适合作为 PPO 验证/测试？
7. 使用所有年份后，是否能显著增加 swfac 水分胁迫样本？
8. 下一阶段应该做 all-year offline schedule search，还是先做品种参数校正？

---

## 5. 本阶段不要做的事情

1. 不要继续直接训练 PPO。
2. 不要做 rainfall-scaling。
3. 不要训练无 action safety PPO。
4. 不要修改 `my_data/` 原始文件。
5. 不要覆盖 006_08 到 006_14 的结果。
6. 不要删除原有 observed-year 实验结果。
7. 不要把 2020–2023 气象数据直接当作最终 RL 结果。
8. 不要假设所有年份都可用，必须先检查文件完整性。
9. 不要只筛选降雨少的年份，还要用 DSSAT/gym-DSSAT 输出 swfac/nstres 诊断。
10. 不要跳过品种参数校正/验证路线说明。

---

## 6. 输入文件和目录

优先检查以下位置：

```text
my_data/
weather_clean_qc/
Leave_One_experiments/wth_generated_qc/
Leave_One_experiments/year_classification/
data/observed_phenology_dates_standardized.csv
data/
experiments/ppo_observed_years/
docs/
```

重点查找：

```text
*.WTH
*.wth
weather_value_range_issues.csv
zero_growing_season_rain_years.csv
observed_phenology_rainfall_rank_by_station.csv
observed_phenology_dates_standardized.csv
```

如果路径不同，请搜索关键词：

```text
WTH
weather
wth_generated
year_classification
rainfall_rank
phenology
station
2020
2021
2022
2023
```

---

## 7. 输出目录

本阶段所有结果保存到：

```text
Leave_One_experiments/all_year_weather_calibration_validation/
```

建议目录结构：

```text
Leave_One_experiments/all_year_weather_calibration_validation/
  configs/
  weather_inventory/
  cultivar_calibration_plan/
  stress_diagnostics/
  scenario_pool/
  daily_outputs/
  evaluation/
  figures/
  reports/
```

报告保存到：

```text
docs/2026-06-06_all_year_weather_calibration_validation_report.md
docs/2026-06-06_all_year_weather_calibration_validation_report.pptx
```

---

## 8. 第一步：建立全站点气象年份 inventory

新增或更新：

```text
src/build_weather_year_inventory.py
```

扫描所有可用 WTH / 气象文件，输出：

```text
Leave_One_experiments/all_year_weather_calibration_validation/weather_inventory/weather_year_inventory.csv
```

字段至少包括：

```text
station_code
station_name
weather_file
weather_source_dir
year
date_start
date_end
n_days
has_full_year
has_growing_season
missing_days_count
missing_days_ratio
growing_season_rain
annual_rain
tmean_mean
tmax_max
tmin_min
srad_mean
weather_qc_status
is_observed_experiment_year
is_2020_2023
notes
```

要求：

1. 必须识别每个站点所有可用年份；
2. 必须单独标记 2020、2021、2022、2023；
3. 必须检查是否覆盖玉米生育期；
4. 必须检查降雨、温度、辐射是否有明显异常；
5. 必须把原来的 observed experiment years 单独标记出来；
6. 必须输出每个站点可用年份数量。

同时输出：

```text
Leave_One_experiments/all_year_weather_calibration_validation/weather_inventory/weather_year_inventory_summary.md
```

---

## 9. 第二步：检查 2020–2023 是否适合品种校正/验证

输出：

```text
Leave_One_experiments/all_year_weather_calibration_validation/cultivar_calibration_plan/weather_2020_2023_availability.csv
```

字段至少包括：

```text
station_code
station_name
year
weather_file
available
has_full_growing_season
weather_qc_status
growing_season_rain
annual_rain
notes
```

然后生成：

```text
Leave_One_experiments/all_year_weather_calibration_validation/cultivar_calibration_plan/cultivar_calibration_validation_plan.md
```

必须说明：

1. 2020–2023 每个站点哪些年份可用；
2. 哪些年份适合 calibration；
3. 哪些年份适合 validation；
4. 如果某站点 2020–2023 不完整，应该如何处理；
5. 是否需要用 2020–2023 的气象重建 X-file / MZX；
6. 品种参数校正时不要混入 PPO action 影响；
7. 校正品种参数时应使用固定管理方案；
8. 校正后再进入 all-year weather scenario PPO。

建议初步划分：

```text
calibration_years:
  2020, 2021

validation_years:
  2022, 2023
```

如果数据缺失，则按实际可用性调整，并在报告中说明。

---

## 10. 第三步：生成或确认 2020–2023 DSSAT/gym-DSSAT 输入方案

如果已有 2020–2023 WTH 文件，则不要重复生成。

如果没有，但可从 cleaned weather 数据生成，则新增或更新：

```text
src/generate_2020_2023_wth_inputs.py
```

如果需要创建新的 X-file / template 配置，请保存到专门输出目录，不要修改 `my_data/` 原始模板：

```text
Leave_One_experiments/all_year_weather_calibration_validation/rendered_inputs/
```

要求：

1. 使用当前每个站点的品种作为初始品种参数；
2. 不要直接覆盖现有模板；
3. 清楚记录每个年份对应哪个 WTH；
4. 如果需要 PDATE，先使用已有 observed phenology 或合理规则；
5. 如果 2020–2023 没有实测 phenology，必须说明品种校正还需要哪些观测数据支持。

---

## 11. 第四步：全年份 fixed management stress diagnostic

这是本阶段最重要的前置诊断。

对所有可用年份运行固定管理方案，不训练 PPO。

至少使用以下固定方案：

```text
T0_null_zero:
  irrigation = 0
  N = 0

T1_N_only_medium:
  irrigation = 0
  N total = 150 kg/ha
  N events = DAP 1, 30, 60

T2_N_medium_I_low:
  irrigation total = 60 mm
  irrigation events = DAP 35, 65
  N total = 150 kg/ha
  N events = DAP 1, 30, 60

T3_N_medium_I_mid:
  irrigation total = 120 mm
  irrigation events = DAP 25, 50, 75
  N total = 150 kg/ha
  N events = DAP 1, 30, 60

T4_N_high_no_I:
  irrigation = 0
  N total = 225 kg/ha
  N events = DAP 1, 30, 60
```

新增或更新：

```text
src/run_all_year_fixed_management_stress_diagnostic.py
```

输出：

```text
Leave_One_experiments/all_year_weather_calibration_validation/stress_diagnostics/all_year_fixed_management_stress_summary.csv
```

字段至少包括：

```text
station_code
station_name
year
treatment_id
weather_file
run_status
episode_completed
final_grnwt
final_topwt
final_xlai
total_irrigation
total_n
profit_default
profit_low_water_cost
growing_season_rain
annual_rain
mean_swfac
max_swfac
swfac_stress_days_gt_0p05
swfac_stress_days_gt_0p10
mean_nstres
max_nstres
nstres_days_gt_0p05
yield_gain_from_irrigation_at_same_N
profit_gain_from_irrigation_default
profit_gain_from_irrigation_low_water_cost
daily_csv_path
notes
```

profit 计算沿用：

```text
profit_default = 0.01 * final_grnwt - 0.5 * total_irrigation - 0.25 * total_n
profit_low_water_cost = 0.01 * final_grnwt - 0.1 * total_irrigation - 0.25 * total_n
```

---

## 12. 第五步：筛选 all-year weather scenario pool

基于第四步结果，输出：

```text
Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/all_year_weather_scenario_pool.csv
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
nitrogen_responsive
growing_season_rain
annual_rain
swfac_stress_days_gt_0p05
max_swfac
mean_swfac
nstres_days_gt_0p05
max_nstres
mean_nstres
yield_gain_from_irrigation_at_same_N
profit_gain_from_irrigation_low_water_cost
recommended_for_calibration
recommended_for_validation
recommended_for_ppo_train
recommended_for_ppo_eval
notes
```

建议 scenario_type 包括：

```text
wet_year
normal_year
dry_year
water_stress_year
nitrogen_stress_year
irrigation_responsive_year
low_response_year
weather_qc_problem
```

筛选规则建议：

```text
has_water_stress:
  swfac_stress_days_gt_0p05 > 0
  或 max_swfac > 0.05

irrigation_responsive:
  同等 N 下加灌溉 final_grnwt 提高 >= 200 kg/ha
  或提高 >= 3%
  且 profit_low_water_cost 不明显变差

recommended_for_ppo_train:
  weather_qc_status ok
  run_status ok
  且包含 water_stress_year / nitrogen_stress_year / irrigation_responsive_year / normal_year 的混合

recommended_for_ppo_eval:
  不用于 train 的年份
  覆盖 dry / normal / wet 类型
```

同时生成：

```text
Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/scenario_pool_summary.md
```

必须说明：

1. 每个站点一共有多少可用年份；
2. 有多少水分胁迫年份；
3. 有多少氮胁迫年份；
4. 有多少灌溉响应年份；
5. 2020–2023 在 scenario pool 中的位置；
6. 哪些年份建议用于 calibration；
7. 哪些年份建议用于 validation；
8. 哪些年份建议用于 PPO train/eval。

---

## 13. 第六步：制定新的训练/验证路线

生成：

```text
Leave_One_experiments/all_year_weather_calibration_validation/evaluation/new_experiment_route_after_group_meeting.md
```

必须明确提出新的实验路线：

### 阶段 A：品种参数校正/验证

```text
使用 2020–2023 年气象数据；
先用固定管理方案；
不引入 PPO；
目标是校正/验证当前站点品种参数；
calibration/validation 年份按实际可用性划分。
```

### 阶段 B：all-year weather scenario stress pool

```text
使用所有可用年份；
按 water stress / nitrogen stress / irrigation response / wet-normal-dry 分类；
构建 PPO 训练和评估年份集合。
```

### 阶段 C：重新训练 expert prior / imitation prior

```text
基于 all-year scenario pool 重新做 offline schedule search；
生成更多 expert trajectories；
尤其补充 water-stress / irrigation-responsive years；
重新训练 BC/RF/two-stage prior。
```

### 阶段 D：PPO 训练

```text
只在 scenario pool 建立后执行；
继续使用 action safety；
可以从 augmented expert prior 初始化；
不要先做 unrestricted PPO。
```

---

## 14. 第七步：给出下一阶段建议

如果 scenario pool 中找到至少：

```text
>= 3 个 water_stress_year
或 >= 3 个 irrigation_responsive_year
```

则报告建议下一阶段为：

```text
006_16_all_year_offline_schedule_search_and_imitation_prior
```

如果没有找到足够水分胁迫年份，则报告建议：

```text
006_16_cultivar_calibration_and_nitrogen_management_focus
```

并说明：

```text
当前所有可用年份仍以氮限制为主，灌溉优化不应作为主线。
```

本阶段只写建议，不要自动创建 006_16，除非我明确要求。

---

## 15. 图表输出

至少生成：

```text
weather_year_inventory_by_station.png
growing_season_rain_by_year_station.png
swfac_stress_days_by_year_station.png
nstres_days_by_year_station.png
yield_gain_from_irrigation_by_year_station.png
scenario_type_heatmap.png
calibration_validation_years_2020_2023.png
ppo_train_eval_scenario_pool.png
```

保存到：

```text
Leave_One_experiments/all_year_weather_calibration_validation/figures/
```

---

## 16. 报告要求

报告必须说明：

1. 为什么要从 observed-year-only 改成 all-year weather scenario；
2. 2020–2023 年气象数据是否齐全；
3. 2020–2023 如何用于品种校正/验证；
4. 所有可用年份 inventory；
5. 哪些年份有水分胁迫；
6. 哪些年份有氮胁迫；
7. 哪些年份有灌溉响应；
8. 使用所有年份后是否增加了 swfac 样本；
9. 新的 PPO 训练/验证年份建议；
10. 是否建议下一阶段做 all-year offline schedule search；
11. 是否建议继续聚焦氮管理；
12. 是否需要返回校正品种参数。

---

## 17. 报告输出

生成 Markdown 报告：

```text
docs/2026-06-06_all_year_weather_calibration_validation_report.md
```

生成 PPT：

```text
docs/2026-06-06_all_year_weather_calibration_validation_report.pptx
```

并复制一份到：

```text
Leave_One_experiments/all_year_weather_calibration_validation/reports/
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
git add prompts/006_15_use_2020_2023_weather_and_all_available_years_for_calibration_validation_and_PPO.md
git add src/
git add experiments/ppo_observed_years/
git add Leave_One_experiments/all_year_weather_calibration_validation/configs/
git add Leave_One_experiments/all_year_weather_calibration_validation/weather_inventory/
git add Leave_One_experiments/all_year_weather_calibration_validation/cultivar_calibration_plan/
git add Leave_One_experiments/all_year_weather_calibration_validation/stress_diagnostics/
git add Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/
git add Leave_One_experiments/all_year_weather_calibration_validation/evaluation/
git add Leave_One_experiments/all_year_weather_calibration_validation/figures/
git add Leave_One_experiments/all_year_weather_calibration_validation/reports/
git add docs/2026-06-06_all_year_weather_calibration_validation_report.md
git add docs/2026-06-06_all_year_weather_calibration_validation_report.pptx
git commit -m "Build all-year weather scenario pool for calibration validation and PPO"
```

注意：不要默认 commit 大量 daily_outputs；不要默认 commit 大模型；不要默认 commit tensorboard；不要强行 push。

---

## 19. 完成后请汇报

完成后请汇报：

1. 每个站点有哪些可用气象年份；
2. 2020–2023 年是否齐全；
3. 哪些年份适合品种 calibration；
4. 哪些年份适合 validation；
5. 所有年份中哪些有水分胁迫；
6. 哪些有灌溉响应；
7. 是否比 observed-year-only 显著增加 swfac 样本；
8. 推荐的 PPO train/eval 年份集合；
9. 下一阶段应该是 all-year offline schedule search，还是先做品种校正；
10. 是否仍建议聚焦氮管理，还是可以重新纳入灌溉优化。