# 给 Codex 的完整 Prompt

你现在需要帮助我完成一个农业强化学习实验框架的第一阶段工作。

当前研究目标是：

> 基于 gym-DSSAT / DSSAT 环境，对五个中国生态站点进行玉米水氮管理优化。现在已经确认海伦站在降雨缩放 0.6 的情况下存在土壤水分胁迫和灌溉优化潜力。下一步需要稳定五个站点的水氮优化策略。

本阶段任务包括三件事：

1. 判断每个站点可用年份中的年份类型：干旱年、正常年、湿润年；
2. 每个站点分别训练三套策略：干旱年训练、正常年训练、湿润年训练；
3. 对每个站点选择一套“最稳定策略”，作为后续跨站点泛化测试的基础策略。

请不要急着直接训练 PPO。先把气象数据、年份分类、WTH 文件、训练配置、验证结果整理清楚。

---

# 一、站点与文件说明

五个生态站点及代码如下：

| 站点中文名 | 站点代码 |
| ----- | ---- |
| 海伦    | HLA  |
| 沈阳    | SYA  |
| 栾城    | LCA  |
| 禹城    | YCA  |
| 封丘    | FQA  |

所有站点相关数据都在项目目录下的：

```bash
my_data/
```

管理数据是对应站点名称的 `.jinja2` 模板。

土壤数据是对应站点名称的 `.SOL` 文件。

品种参数文件是：

```bash
MZCER048.CUL
```

当前已经有一些可用于训练的 `CN站点年份.WTH` 文件，但现在需要重新系统整理气象数据，并为每个站点、每个可用年份生成对应的 `.WTH` 文件。

---

# 二、气象数据整理要求

目前站点的原始气象数据还没有整理，请先整理气象数据。

## 1. 气象变量对应关系

DSSAT 所需变量如下：

| DSSAT变量 | 含义             | 原始数据来源变量    |
| ------- | -------------- | ----------- |
| SRAD    | 日总辐射，MJ/m²/day | 总辐射总量，MJ/m² |
| TMAX    | 日最高气温，℃        | 日最大值，℃      |
| TMIN    | 日最低气温，℃        | 日最小值，℃      |
| RAIN    | 日降雨量，mm/day    | 20-20合计，mm  |

只使用 **2000年以后的数据**，即：

```text
year >= 2000
```

---

## 2. 原始气象文件位置

HL、FQ、LC、YC 四个站点：

SRAD 在：

```bash
my_data/D32.xls
```

TMAX、TMIN 在：

```bash
my_data/T2.xls
```

RAIN 在：

```bash
my_data/HLLCYCFQ降雨数据.xls
```

沈阳站 SY：

SRAD 在：

```bash
my_data/SYCS自动气象站观测逐日太阳辐射总量及其累计值.xls
```

TMAX、TMIN 在：

```bash
my_data/SYCS自动气象站观测逐日气温.xls
```

RAIN 在：

```bash
my_data/SYCS人工大气观测降雨蒸发能见度日值.xls
```

注意：

这些 Excel 表格里的数字很多都是**文本格式**，读取后必须先转换成数值型。

RAIN 的原始变量名可能是：

```text
20-20合计(mm)
```

或者类似形式。请在代码中做字段名模糊匹配，优先寻找包含以下关键词的列：

```text
20-20
合计
RAIN
降雨
降水
```

读取后统一命名为：

```text
RAIN
```

单位为：

```text
mm/day
```

---

## 3. 缺失值处理

缺失值处理规则：

> 使用对应站点、对应月份的月平均值替代。

例如：

某站点 2007-06-15 的 SRAD 缺失，则使用该站点所有年份中 6 月 SRAD 的平均值替代。

要求：

1. 对 SRAD、TMAX、TMIN、RAIN 分别处理；
2. 先把文本型数字转换为 numeric；
3. 对无法转换的异常值设为 NaN；
4. 再用月平均值填补；
5. 如果某个月全部缺失，则用该站点全年平均值填补；
6. 最后检查每个站点每年是否仍存在缺失值；
7. 所有缺失值替代数量要保存日志，方便后续检查。

---

## 4. 整理后的气象数据保存

请把整理后的每日气象数据单独保存，方便后续调试。

整理后的气象数据文件夹名称必须是：

```bash
weather_clean/
```

不要叫：

```bash
outputs/weather_cleaned/
```

每个站点保存一个 CSV：

```bash
weather_clean/HLA_weather_cleaned.csv
weather_clean/SYA_weather_cleaned.csv
weather_clean/LCA_weather_cleaned.csv
weather_clean/YCA_weather_cleaned.csv
weather_clean/FQA_weather_cleaned.csv
```

同时生成一份总表：

```bash
weather_clean/all_sites_weather_cleaned.csv
```

每个 CSV 至少包含以下字段：

```text
station
date
year
month
day
doy
SRAD
TMAX
TMIN
RAIN
```

---

# 三、判断每个站点的干旱年、正常年、湿润年

请基于每个站点整理后的气象数据，对每个站点单独判断年份类型。

不要强行让所有站点使用相同年份。

原因：

> 同一年在不同站点可能气候状态不同。例如某一年在海伦可能偏旱，但在栾城不一定偏旱。

---

## 1. 年份类型判断方法

现在已经明确有 RAIN 数据，所以年份类型判断必须优先使用**生育期降雨量**。

优先使用生育期气象，而不是全年气象。

如果 jinja2 模板中能读出种植日期和成熟日期，则使用：

```text
种植日期 至 成熟日期
```

如果暂时无法自动读取成熟日期，则先使用一个固定玉米生育期窗口，例如：

```text
5月1日 到 9月30日
```

或者根据每个站点 jinja2 模板中的种植日期，设置：

```text
种植日期后 150 天
```

请在代码注释、日志和 PPT 记录中说明实际采用的方法。

---

## 2. 干湿分类指标

对每个站点、每一年计算：

```text
growing_season_rain = 生育期总降雨量
```

然后在该站点内部排序：

```text
RAIN 最少的年份 = 干旱年
RAIN 中间的年份 = 正常年
RAIN 最多的年份 = 湿润年
```

如果可用年份很多，不要只机械选最小、中位数、最大值，也要输出完整排序表，方便人工检查。

同时请计算辅助气候指标：

```text
mean_RAIN
sum_RAIN
mean_SRAD
sum_SRAD
mean_TMAX
mean_TMIN
heat_days_TMAX_gt_30
```

如果可以通过 gym-DSSAT 跑 NullAgent 或固定管理策略，则进一步计算每年：

```text
mean_SWFAC
mean_NSTRES
final_TOPWT
final_GRNWT
```

然后辅助判断：

| 类型  | 特征                 |
| --- | ------------------ |
| 干旱年 | 生育期降雨量较少，可能水分胁迫更明显 |
| 正常年 | 生育期降雨量中等，胁迫中等      |
| 湿润年 | 生育期降雨量较多，水分胁迫较弱    |

注意：在 gym-DSSAT 里，SWFAC / NSTRES 的含义需要确认。不要在代码里硬编码错误解释。只做数值对比和相对排序。

---

## 3. 年份分类结果保存

请输出每个站点的年份统计表：

```bash
Leave_One_experiments/year_classification/site_year_climate_summary.csv
```

字段包括：

```text
station
year
growing_season_start
growing_season_end
growing_season_rain
mean_RAIN
sum_RAIN
mean_SRAD
sum_SRAD
mean_TMAX
mean_TMIN
heat_days_TMAX_gt_30
year_type
classification_reason
```

再输出最终每个站点选中的三类年份：

```bash
Leave_One_experiments/year_classification/selected_years.csv
```

字段包括：

```text
station
dry_year
normal_year
wet_year
dry_year_rain
normal_year_rain
wet_year_rain
notes
```

---

# 四、生成 DSSAT WTH 文件

请根据整理后的每日气象数据，为每个站点、每个可用年份生成 DSSAT `.WTH` 文件。

WTH 文件必须包含：

```text
SRAD
TMAX
TMIN
RAIN
```

输出目录：

```bash
Leave_One_experiments/wth_generated/{station}/
```

例如：

```bash
Leave_One_experiments/wth_generated/HLA/HLA2007.WTH
Leave_One_experiments/wth_generated/SYA/SYA2007.WTH
```

如果 DSSAT 或 gym-DSSAT 对 WTH 文件命名有长度或格式要求，请按照当前项目中已有的 `CN站点年份.WTH` 文件格式保持一致。

请先读取现有可用的 WTH 文件作为模板，保证格式兼容 gym-DSSAT。

---

# 五、训练与验证实验总目录

保存训练、验证、模型、日志、图表、日值结果的大文件夹统一命名为：

```bash
Leave_One_experiments/
```

请不要把主要训练结果保存到普通的 `outputs/` 目录下面。

推荐目录结构如下：

```bash
Leave_One_experiments/
  year_classification/
    site_year_climate_summary.csv
    selected_years.csv

  wth_generated/
    HLA/
    SYA/
    LCA/
    YCA/
    FQA/

  models/
    HLA/
    SYA/
    LCA/
    YCA/
    FQA/

  logs/
    HLA/
    SYA/
    LCA/
    YCA/
    FQA/

  evaluation/
    HLA/
    SYA/
    LCA/
    YCA/
    FQA/

  daily_outputs/
    HLA/
    SYA/
    LCA/
    YCA/
    FQA/

  figures/
    HLA/
    SYA/
    LCA/
    YCA/
    FQA/

  strategy_selection/
    best_policy_by_site.csv

  reports/
    01_五站点年份分类与站点内策略稳定性实验记录.pptx
```

代码可以放在：

```bash
src/
experiments/
```

但所有实验结果统一进入：

```bash
Leave_One_experiments/
```

---

# 六、训练与验证代码组织要求

每个站点的训练最好建立单独的训练、验证、绘图代码，不要都在一个代码里面改来改去，这样后续更方便调试。

推荐目录结构如下：

```bash
experiments/
  HLA/
    train_HLA_dry.py
    train_HLA_normal.py
    train_HLA_wet.py
    evaluate_HLA.py
    plot_HLA.py
    config_HLA.yaml

  SYA/
    train_SYA_dry.py
    train_SYA_normal.py
    train_SYA_wet.py
    evaluate_SYA.py
    plot_SYA.py
    config_SYA.yaml

  LCA/
    train_LCA_dry.py
    train_LCA_normal.py
    train_LCA_wet.py
    evaluate_LCA.py
    plot_LCA.py
    config_LCA.yaml

  YCA/
    train_YCA_dry.py
    train_YCA_normal.py
    train_YCA_wet.py
    evaluate_YCA.py
    plot_YCA.py
    config_YCA.yaml

  FQA/
    train_FQA_dry.py
    train_FQA_normal.py
    train_FQA_wet.py
    evaluate_FQA.py
    plot_FQA.py
    config_FQA.yaml
```

也可以使用一个通用核心模块，例如：

```bash
src/
  weather_preprocess.py
  wth_writer.py
  year_classifier.py
  train_policy.py
  evaluate_policy.py
  plot_results.py
  utils.py
```

然后每个站点脚本只调用通用模块。

核心原则：

> 通用逻辑放在 src/，每个站点保留独立入口脚本，方便后续单独调试某个站点。

---

# 七、每个站点的三套策略

每个站点做三套训练策略。

以海伦 HLA 为例。

假设年份分类为：

```text
dry_year = 2007
normal_year = 2009
wet_year = 2011
```

则训练三套策略。

## 策略1：干旱年训练

```text
训练：2007
验证：2009 + 2011
```

输出模型：

```bash
Leave_One_experiments/models/HLA/HLA_policy_train_dry_seed0.zip
```

## 策略2：正常年训练

```text
训练：2009
验证：2007 + 2011
```

输出模型：

```bash
Leave_One_experiments/models/HLA/HLA_policy_train_normal_seed0.zip
```

## 策略3：湿润年训练

```text
训练：2011
验证：2007 + 2009
```

输出模型：

```bash
Leave_One_experiments/models/HLA/HLA_policy_train_wet_seed0.zip
```

其他站点同理。

---

# 八、随机种子要求

由于训练较复杂，暂时先只使用一个随机种子。

默认：

```text
seed = 0
```

所有代码中保留 seed 参数，方便后续扩展，但本阶段不要默认运行多个 seed。

也就是说，当前阶段运行：

```text
5个站点 × 3种训练年份 × 1个seed
```

后续根据实际算力，再考虑扩展为：

```text
5个站点 × 3种训练年份 × 多个seed
```

请在 README、日志和 PPT 里明确说明：

> 本阶段暂时使用单随机种子进行流程验证和策略稳定性初筛，后续可扩展到多随机种子鲁棒性分析。

---

# 九、训练要求

每个策略至少保存以下内容：

```bash
Leave_One_experiments/models/{station}/
Leave_One_experiments/logs/{station}/
Leave_One_experiments/evaluation/{station}/
Leave_One_experiments/daily_outputs/{station}/
Leave_One_experiments/figures/{station}/
```

每次训练需要记录：

```text
station
train_year
train_year_type
validation_years
PPO hyperparameters
reward function version
state variables
action variables
random seed
total_timesteps
final model path
training start time
training end time
```

训练前先用 NullAgent 或固定策略跑每个站点的 dry、normal、wet 年份，确认 gym-DSSAT 环境没有卡死或报错。

---

# 十、验证指标

每个策略在两个验证年份上评估。

至少输出以下指标：

```text
station
policy_name
train_year
train_year_type
eval_year
eval_year_type
seed
final_yield_grnwt
final_topwt
total_irrigation
total_n_fertilizer
mean_swfac
mean_nstres
mean_reward
sum_reward
episode_length
```

如果能计算经济收益，则额外输出：

```text
gross_return
irrigation_cost
fertilizer_cost
net_profit
```

如果能计算水氮利用效率，则额外输出：

```text
WUE = yield / total_irrigation
NUE = yield / total_n_fertilizer
```

注意处理除以 0 的情况。

---

# 十一、非常重要：每一次实验都必须保存日值数据

这是本任务最重要的要求之一。

每一次训练后验证时，每个策略、每个验证年份都必须保存完整的日值数据。

不仅要保存最后的产量、灌溉量、施氮量，还要保存逐日过程。

日值数据保存目录：

```bash
Leave_One_experiments/daily_outputs/{station}/
```

建议文件命名：

```bash
Leave_One_experiments/daily_outputs/HLA/HLA_policy_train_dry_eval_normal_2009_seed0_daily.csv
Leave_One_experiments/daily_outputs/HLA/HLA_policy_train_dry_eval_wet_2011_seed0_daily.csv
```

每个 daily csv 至少包含以下字段：

```text
station
policy_name
train_year
train_year_type
eval_year
eval_year_type
seed
date
year
doy
dap
topwt
grnwt
xlai
totir
tofer
swfac
nstres
reward
real_action_amir
real_action_anfer
normalized_action_amir
normalized_action_anfer
```

其中必须包括：

```text
dap
topwt
grnwt
xlai
totir
tofer
swfac
nstres
reward
real_action_amir
real_action_anfer
normalized_action_amir
normalized_action_anfer
```

如果当前环境中变量名不是 `tofer`，而是其他累计施肥变量，请在代码中检查并统一输出为：

```text
tofer
```

如果环境中没有 `tofer`，但可以通过每日 `real_action_anfer` 累加得到，则计算：

```text
tofer = cumulative_sum(real_action_anfer)
```

同理：

```text
totir = cumulative_sum(real_action_amir)
```

如果环境原始 observation 中已经有 `totir`，则同时保存原始 `totir_raw` 和修正后的 `totir`，避免混淆。

---

# 十二、动作变量保存要求

水氮联合优化时，动作至少包括：

```text
amir
anfer
```

其中：

| 变量    | 含义   |
| ----- | ---- |
| amir  | 灌溉动作 |
| anfer | 施肥动作 |

请同时保存：

```text
real_action_amir
real_action_anfer
normalized_action_amir
normalized_action_anfer
```

解释：

| 字段                      | 含义                           |
| ----------------------- | ---------------------------- |
| real_action_amir        | 反归一化后的真实灌溉量                  |
| real_action_anfer       | 反归一化后的真实施肥量                  |
| normalized_action_amir  | PPO 输出或 wrapper 内部使用的归一化灌溉动作 |
| normalized_action_anfer | PPO 输出或 wrapper 内部使用的归一化施肥动作 |

如果当前 wrapper 里没有直接保存 normalized action，请修改 wrapper 或 evaluation 代码，把模型输出动作和实际传入 DSSAT 的动作都记录下来。

---

# 十三、每一次验证实验都必须生成响应折线图

每一次验证实验都要生成对应的响应折线图。

图保存目录：

```bash
Leave_One_experiments/figures/{station}/daily_response/
```

每个验证年份至少生成以下图。

---

## 1. DAP - SWFAC - 灌溉响应图

横轴：

```text
dap
```

纵轴：

```text
swfac
real_action_amir
reward
```

建议画成多轴或分图，但不要丢失任何一个变量。

文件名示例：

```bash
HLA_policy_train_dry_eval_normal_2009_seed0_dap_swfac_irrigation_reward.png
```

---

## 2. DAP - NSTRES - 施肥响应图

横轴：

```text
dap
```

纵轴：

```text
nstres
real_action_anfer
reward
```

文件名示例：

```bash
HLA_policy_train_dry_eval_normal_2009_seed0_dap_nstres_fertilization_reward.png
```

---

## 3. 作物生长过程图

横轴：

```text
dap
```

纵轴：

```text
topwt
grnwt
xlai
```

文件名示例：

```bash
HLA_policy_train_dry_eval_normal_2009_seed0_crop_growth_timeseries.png
```

---

## 4. 累计水氮投入图

横轴：

```text
dap
```

纵轴：

```text
totir
tofer
```

文件名示例：

```bash
HLA_policy_train_dry_eval_normal_2009_seed0_cumulative_water_nitrogen.png
```

---

# 十四、每个站点最优策略的图要更加完整

每个站点选出最稳定策略后，要单独生成一个 `best_policy` 图表文件夹：

```bash
Leave_One_experiments/figures/{station}/best_policy/
```

至少包括：

```text
1. best_policy_daily_actions_by_year.png
2. best_policy_swfac_nstres_by_year.png
3. best_policy_topwt_grnwt_xlai_by_year.png
4. best_policy_reward_by_year.png
5. best_policy_cumulative_irrigation_fertilization_by_year.png
```

这些图要对比该最优策略在不同验证年份下的表现。

例如海伦最优策略是 `HLA_policy_train_dry`，验证年份是正常年和湿润年，则图中应同时展示：

```text
eval_normal_year
eval_wet_year
```

---

# 十五、评价汇总文件也要关联日值数据路径

每一次验证结果汇总时，除了保存最终指标，还要保存对应的 daily csv 路径和 figure 路径。

评价汇总文件保存为：

```bash
Leave_One_experiments/evaluation/all_policy_evaluation_summary.csv
```

字段至少包括：

```text
station
policy_name
train_year
train_year_type
eval_year
eval_year_type
seed
final_yield_grnwt
final_topwt
total_irrigation
total_n_fertilizer
mean_swfac
mean_nstres
mean_reward
sum_reward
episode_length
daily_csv_path
figure_dir
```

每个站点也保存一份：

```bash
Leave_One_experiments/evaluation/{station}/{station}_policy_evaluation_summary.csv
```

---

# 十六、选择每个站点“最稳定策略”

不要只选择产量最高的策略。

每个站点的最稳定策略应该综合考虑：

1. 验证年份平均产量较高；
2. 验证年份之间波动较小；
3. 灌溉量不要过大；
4. 施氮量不要过大；
5. reward 较高；
6. 在干旱年、正常年、湿润年之间表现不过度崩溃。

当前阶段每个策略只有一个 seed，因此稳定性主要看：

```text
同一策略在两个验证年份上的平均表现和波动
```

暂时不要把 seed 间波动纳入评分。

建议先做简单标准化评分：

```text
score =
  + normalized_mean_yield
  + normalized_mean_reward
  - normalized_yield_std
  - normalized_total_irrigation
  - normalized_total_n_fertilizer
```

其中每个指标在同一站点三套策略之间归一化到 0-1。

如果暂时没有经济价格参数，就先不用净收益。

请在结果说明中写清楚：

> 当前 stability_score 是基于跨年份验证表现计算的，不包含多随机种子不确定性。后续如果算力允许，可增加多 seed 训练，并把 seed 间方差纳入稳定性评分。

最终输出：

```bash
Leave_One_experiments/strategy_selection/best_policy_by_site.csv
```

字段包括：

```text
station
best_policy_name
best_train_year
best_train_year_type
validation_years
mean_yield
std_yield
mean_reward
std_reward
mean_irrigation
mean_n_fertilizer
stability_score
reason
model_path
```

---

# 十七、绘图总体要求

每个站点单独绘图。

建议输出：

```bash
Leave_One_experiments/figures/{station}/
```

每个站点至少包含：

1. 年份气候分类图；
2. 三套策略验证产量对比图；
3. 三套策略验证灌溉量对比图；
4. 三套策略验证施氮量对比图；
5. 三套策略稳定性评分图；
6. 每一次验证实验的 DAP-SWFAC-灌溉-reward 响应图；
7. 每一次验证实验的 DAP-NSTRES-施肥-reward 响应图；
8. 每一次验证实验的 TOPWT / GRNWT / XLAI 生长过程图；
9. 每一次验证实验的累计灌溉 / 累计施肥图；
10. 每个站点最优策略在不同年份的每日灌溉/施肥动作图；
11. 每个站点最优策略在不同年份的 SWFAC / NSTRES / TOPWT / GRNWT / XLAI 时间序列图。

注意：

> 不要只画最终结果柱状图。必须保存逐日过程数据和逐日响应图，否则后续无法分析 PPO 为什么这样灌溉和施肥。

---

# 十八、PPT 记录要求

请将所做的一切实验和尝试记录到 PPT 中保存下来。

PPT 文件名必须使用中文，并且在 PPT 名称前加上序号。

PPT 保存路径为：

```bash
Leave_One_experiments/reports/01_五站点年份分类与站点内策略稳定性实验记录.pptx
```

PPT 至少包括：

1. 任务目标；
2. 数据来源；
3. SRAD、TMAX、TMIN、RAIN 数据来源；
4. 气象数据整理方法；
5. 文本型数字转 numeric 的处理；
6. 缺失值处理方法；
7. 每个站点可用年份；
8. 生育期降雨量排序结果；
9. 每个站点干旱、正常、湿润年份判定结果；
10. 每个站点干旱、正常、湿润年份选择依据；
11. WTH 文件生成结果；
12. 每个站点三套策略的训练-验证年份组合；
13. 每个站点三套策略训练设置；
14. 每个策略验证年份的最终结果；
15. 每个策略验证年份的日值过程图；
16. DAP-SWFAC-灌溉-reward 响应关系；
17. DAP-NSTRES-施肥-reward 响应关系；
18. TOPWT、GRNWT、XLAI 生长过程；
19. 累计灌溉量和累计施肥量；
20. 每个站点最稳定策略选择结果；
21. 每个站点最稳定策略的日值响应图；
22. 当前单 seed 的局限性；
23. 当前发现的问题；
24. 每一次代码修改、实验失败、报错、修复方法；
25. 后续多 seed 扩展建议；
26. 下一步建议。

每一次代码修改、实验失败、报错、修复方法都要简要记录到 PPT 或日志中，方便后续组会汇报。

---

# 十九、GitHub 备份要求

调试好的代码和结果需要备份到 GitHub。

请完成以下事情：

1. 检查当前项目是否已经是 Git 仓库；
2. 如果不是，请初始化 Git；
3. 添加合理的 `.gitignore`；
4. 不要上传过大的模型文件、临时文件、缓存文件；
5. 代码、配置文件、核心结果 CSV、PPT 可以上传；
6. 每完成一个阶段进行一次 commit；
7. 如果远程 GitHub 仓库还没有配置，请先不要强行 push，先告诉我需要配置 remote。

推荐 commit 信息：

```bash
git add .
git commit -m "Add weather preprocessing and WTH generation for five sites"

git add .
git commit -m "Add year classification and leave-one-year experiment configs"

git add .
git commit -m "Add site-specific PPO training and validation scripts"

git add .
git commit -m "Add daily outputs, figures, and strategy selection results"

git add .
git commit -m "Add experiment report PPT for five-site strategy stabilization"
```

---

# 二十、最终交付物清单

请最终确保有以下文件或目录：

```bash
weather_clean/
weather_clean/HLA_weather_cleaned.csv
weather_clean/SYA_weather_cleaned.csv
weather_clean/LCA_weather_cleaned.csv
weather_clean/YCA_weather_cleaned.csv
weather_clean/FQA_weather_cleaned.csv
weather_clean/all_sites_weather_cleaned.csv

Leave_One_experiments/
Leave_One_experiments/year_classification/site_year_climate_summary.csv
Leave_One_experiments/year_classification/selected_years.csv
Leave_One_experiments/wth_generated/
Leave_One_experiments/models/
Leave_One_experiments/logs/
Leave_One_experiments/evaluation/all_policy_evaluation_summary.csv
Leave_One_experiments/evaluation/{station}/{station}_policy_evaluation_summary.csv
Leave_One_experiments/daily_outputs/
Leave_One_experiments/figures/
Leave_One_experiments/strategy_selection/best_policy_by_site.csv
Leave_One_experiments/reports/01_五站点年份分类与站点内策略稳定性实验记录.pptx

src/
experiments/
```

---

# 二十一、建议执行顺序

请按下面顺序执行，不要跳步：

```text
Step 1. 扫描 my_data 文件夹，列出所有 jinja2、SOL、WTH、CUL、xls 文件。

Step 2. 读取 HL/FQ/LC/YC 的 SRAD、TMAX、TMIN、RAIN。
        其中 SRAD 来自 D32.xls；
        TMAX、TMIN 来自 T2.xls；
        RAIN 来自 HLLCYCFQ降雨数据.xls。

Step 3. 读取 SY 的 SRAD、TMAX、TMIN、RAIN。
        SRAD 来自 SYCS自动气象站观测逐日太阳辐射总量及其累计值.xls；
        TMAX、TMIN 来自 SYCS自动气象站观测逐日气温.xls；
        RAIN 来自 SYCS人工大气观测降雨蒸发能见度日值.xls。

Step 4. 将所有文本型数字转换为 numeric。

Step 5. 按站点和月份平均值填补 SRAD、TMAX、TMIN、RAIN 缺失值。

Step 6. 保存整理后的气象数据到 weather_clean/。

Step 7. 根据 cleaned weather 生成每个站点每年的 WTH 文件。

Step 8. 基于生育期 RAIN 判断每个站点的干旱年、正常年、湿润年。

Step 9. 保存 site_year_climate_summary.csv 和 selected_years.csv。

Step 10. 先用 NullAgent 或固定策略跑每个站点的 dry、normal、wet 年份，确认环境不报错、不卡死。

Step 11. 每个站点生成三套训练脚本：
         dry year 训练；
         normal year 训练；
         wet year 训练。

Step 12. 每套策略在另外两个年份验证。

Step 13. 每一次验证都保存完整 daily csv。

Step 14. 每一次验证都生成：
         DAP-SWFAC-灌溉-reward 图；
         DAP-NSTRES-施肥-reward 图；
         TOPWT/GRNWT/XLAI 生长过程图；
         累计水氮投入图。

Step 15. 汇总每个策略在两个验证年份上的 Mean ± SD。

Step 16. 计算每个站点三套策略的 stability_score。

Step 17. 每个站点选择一套最稳定策略。

Step 18. 为每个站点最稳定策略生成 best_policy 图表。

Step 19. 整理所有 CSV、图和结果进 PPT。

Step 20. Git commit 备份代码、核心结果和 PPT。
```

---

# 二十二、如果训练成本太高，请先做小规模调试

请先用以下方式调试流程：

```text
站点：HLA
年份：选出的 dry / normal / wet 三年
seed：0
total_timesteps：较小值
```

确认完整流程跑通后，再扩展到：

```text
5个站点 × 3种训练年份 × 1个seed
```

不要一开始就把所有实验全部开跑，否则很难定位问题。

---

# 二十三、特别强调

请务必注意以下几点：

1. 先整理气象数据，再判断年份类型，不要直接训练；
2. RAIN 已经有原始数据，年份分类必须优先基于生育期降雨量；
3. SRAD、TMAX、TMIN、RAIN 都必须整理；
4. 所有 Excel 中的文本型数字必须转换为 numeric；
5. 缺失值使用站点内对应月份平均值替代；
6. 整理后的气象数据目录叫 `weather_clean/`，不要叫 `outputs/weather_cleaned/`；
7. 训练实验大目录叫 `Leave_One_experiments/`；
8. 每个站点单独判断干旱年、正常年、湿润年；
9. 不要强行所有站点使用同一组年份；
10. 每个站点训练三套策略；
11. 当前阶段只使用一个随机种子 `seed=0`；
12. 所有代码保留多 seed 扩展接口，但不要默认运行多个 seed；
13. 每个站点选择一套最稳定策略；
14. 不要把所有站点代码混在一个大脚本里反复改；
15. 每个站点最好有独立训练、验证、绘图脚本；
16. 所有中间结果都要保存成本地文件，方便检查；
17. 每一次验证实验都必须保存日值数据；
18. 日值数据必须包括：

```text
dap
topwt
grnwt
xlai
totir
tofer
swfac
nstres
reward
real_action_amir
real_action_anfer
normalized_action_amir
normalized_action_anfer
```

19. 每一次验证实验都必须画响应折线图；
20. 每个站点最稳定策略必须单独画 best_policy 图；
21. PPT 中必须记录气象整理、年份分类、训练结果、验证结果、日值响应过程和失败尝试；
22. 调试好的代码和结果及时备份到 GitHub；
23. 如果某一步发现数据缺失、字段名不匹配、WTH 格式不确定，不要跳过，要输出明确的检查报告；
24. 训练前先用 NullAgent 或固定策略跑通每个站点每个年份，确认 gym-DSSAT 环境没有卡死或报错；
25. 不要只保存最终产量，必须保存逐日过程，否则后续无法分析 PPO 为什么这样灌溉和施肥。
