# 004_02：YC WGEN 随机生长季天气 ensemble 验证（按 Wang et al. 2025 路线）

## 一、任务定位

上一阶段已经确认：

```text
1. YC 2004–2013 frozen weather → WGEN 参数拟合方法：PASS_WITH_NOTE
2. 正式 CNYC.CLI：已冻结且 schema/hash 正确
3. Gym-DSSAT + DSSAT WGEN：能够在 random weather 模式下逐日生成天气
4. 同 weather_seed 可复现
5. 不同 weather_seed 能生成不同天气序列
6. 当前 episode 约 116–120 天，因为玉米成熟后仿真结束
```

上一轮对完整 365/366 天 WGEN 路径的调查结论是：

```text
NO_VERIFIED_AUTOMATED_FULL_YEAR_WGEN_PATH
```

这一点 **不再作为 YC random-weather PPO 的 blocker**。

本轮回到 Wang et al. (2025) 的实验思路：

> 在 Gym-DSSAT episode 内由 DSSAT WGEN 生成随机天气 realization，并让强化学习策略在天气不确定性下训练/评估。

本轮只验证：

> **PPO 实际会经历的随机生长季天气是否具有合理的禹城历史气候统计特征。**

不要求先生成完整日历年。

---

# 二、核心研究问题

本轮回答三个问题：

```text
Q1. Gym-DSSAT + WGEN 能否稳定批量产生足够多的 YC random-weather episodes？

Q2. 这些 episode 内的 RAIN / SRAD / TMAX / TMIN 是否与禹城历史同期生长季天气统计特征一致、合理？

Q3. 如果通过，是否可以进入 YC random-weather PPO pilot？
```

---

# 三、严格冻结项

不得修改：

```text
results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv

results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI

scripts/build_dssat_cli.py 的 WGEN 参数定义与算法

wet-day threshold = RAIN > 0.0 mm

current Gym-DSSAT runtime

DSSAT binary

reward function

action space

observation space

YC crop/soil/cultivar/management baseline
```

不得：

```text
重建 DSSAT
重装 WeatherMan
继续追完整全年 WGEN
重新拟合 CLI
用 validation 数据调参数
运行正式长周期 PPO
制作 PPT/PPTX
```

---

# 四、参考路线

主要依据：

```text
Richardson (1981)
Richardson & Wright (1984)
Soltani et al. (2000)
Soltani & Hoogenboom (2007)
Gym-DSSAT random_weather mechanism
Wang et al. (2025)
```

Wang et al. (2025) 的相关思想：

```text
Gym-DSSAT
→ DSSAT WGEN stochastic weather
→ random weather realizations by episode
→ RL under weather uncertainty
```

本轮不要求逐字复制 Wang 的 scenario 设置，而是采用其：

```text
episode-level stochastic weather
```

作为研究框架。

---

# 五、生成规模

生成：

```text
N = 100 independent random-weather episodes
```

建议 weather_seed：

```text
1001–1100
```

所有 episode：

```text
random_weather = True
WTHER = W
same YC CNYC.CLI
same YC crop/soil/cultivar/management configuration
```

本轮：

```text
ppo_seed = NOT_APPLICABLE
```

只生成天气，不训练 PPO。

---

# 六、重要：统一比较窗口

不同 random-weather episode 会因为天气影响作物物候而出现不同成熟日期，因此 episode 长度可能不同。

不能直接比较：

```text
每个 episode 从起点到各自成熟日的累计降雨
```

然后说某个 seed 更干/更湿，因为比较窗口长度不同。

必须建立至少两个统计口径，并严格区分。

## 口径 A：固定共同日期窗口（主验证口径）

选择一个所有历史年份和 synthetic episode 都完整覆盖的固定日历窗口。

优先使用此前已验证过的：

```text
June 1 – September 24
```

但必须先确认：

```text
100 个 synthetic episodes 全部覆盖到 September 24
```

如果有 episode 在 9 月 24 日前结束：

选择：

```text
所有 synthetic episodes 都覆盖的最大共同日期终点
```

并记录选择依据。

### 这个口径用于主气候验证：

```text
RAIN total
wet days
mean rain per wet day
max daily rain
longest dry spell

TMAX mean/SD/extremes
TMIN mean/SD/extremes
SRAD mean/SD/extremes
```

---

## 口径 B：各自完整 crop-season（辅助口径）

记录每个 episode：

```text
simulation_start
simulation_end
episode_days
anthesis_date if available
maturity_date if available
season_total_rain
season_wet_days
season_TMAX_mean
season_TMIN_mean
season_SRAD_mean
```

这个口径仅用于描述：

```text
作物实际经历的 weather exposure
```

不能和固定窗口统计混用。

---

# 七、历史对照数据

必须保持 fitting / validation 分离。

## Fitting reference

```text
2004–2013 observed YC weather
```

对每一年提取与 synthetic **完全相同的固定日期窗口**。

## Independent validation reference

```text
2014–2023 observed YC weather
```

同样提取相同固定日期窗口。

2014–2023：

```text
不得用于 WGEN fitting
不得用于 threshold tuning
不得用于筛 seed
不得用于删除“看起来不好”的 synthetic episodes
```

只用于 independent comparison。

---

# 八、随机 seed 不允许筛选

100 个 seed：

```text
1001–1100
```

全部保留。

禁止：

```text
只保留“合理 seed”
删除偏干/偏湿 seed
重新抽 seed 直到分布好看
```

所有 seed 的结果都必须进入统计。

若出现异常：

```text
保留
标记
解释
```

---

# 九、工程门禁

每个 episode 检查：

```text
DSSAT run success
WGEN active
expected weather_seed injected
continuous dates
no duplicate dates
finite RAIN/SRAD/TMAX/TMIN

RAIN >= 0
SRAD >= 0
TMAX >= TMIN
```

输出：

```text
episode_generation_qc.csv
```

---

# 十、reproducibility 与 diversity

从 100 个 seed 中预先指定：

```text
1001
1025
1050
1075
1100
```

各重复运行一次。

要求：

```text
same weather_seed → daily weather identical
```

同时验证：

```text
different weather_seed → distinct sequences
```

不得临时挑 seed。

---

# 十一、降雨验证（最高优先级）

固定共同窗口内，逐年/逐 episode 计算：

```text
rain_total
wet_days
rain_per_wet_day_mean
rain_per_wet_day_median
max_daily_rain
longest_dry_spell
longest_wet_spell
```

分别汇总：

```text
A. observed fitting years 2004–2013
B. observed validation years 2014–2023
C. synthetic 100 episodes
```

计算：

```text
mean
SD
CV
median
min
max
P05
P10
P25
P75
P90
P95
```

---

# 十二、降雨季节内部结构

固定窗口内按月比较：

```text
June
July
August
September（仅使用共同窗口包含的日期）
```

统计：

```text
monthly rainfall
monthly wet days
monthly rain intensity
monthly max daily rain
```

避免出现：

```text
总降雨合理
但时间分配极不合理
```

---

# 十三、温度验证

固定窗口比较：

```text
TMAX mean
TMAX SD
TMAX P05/P50/P95
TMAX min/max

TMIN mean
TMIN SD
TMIN P05/P50/P95
TMIN min/max
```

额外比较：

```text
daily TMAX-TMIN
hot-tail frequency
cold-tail frequency
```

极端阈值优先用 fitting observed empirical quantiles 做描述，不据 validation 调整生成器。

---

# 十四、SRAD 验证

比较：

```text
mean
SD
P05
P50
P95
min
max
monthly climatology
```

---

# 十五、相关结构

至少比较 observed vs synthetic：

```text
RAIN occurrence vs SRAD
RAIN occurrence vs TMAX
TMAX vs TMIN
TMAX vs SRAD
```

如可行，再计算：

```text
lag-1 autocorrelation:
TMAX
TMIN
SRAD
wet/dry sequence
```

---

# 十六、评价原则

不要用单个 p-value 决定 weather quality。

重点看：

```text
central tendency
variance
distribution overlap
quantiles
seasonality
dry/wet structure
physical plausibility
systematic bias
```

可以辅助使用：

```text
KS
Mann–Whitney
Brown-Forsythe
bootstrap CI
```

但报告中必须强调：

```text
observed years sample size is small
```

---

# 十七、不要要求 synthetic 每个 seed 都“正常”

WGEN 的目的就是产生年际随机性。

合理结果允许出现：

```text
较干 episode
较湿 episode
较热 episode
较凉 episode
```

真正需要警惕的是：

```text
大量 seed 系统性偏离 observed range
明显不合理的极端频率
所有 seed 过于相似
严重月尺度结构错误
明显 shared bias
```

---

# 十八、建议输出图

至少：

```text
1. fixed-window rainfall distribution
2. wet-day distribution
3. longest dry-spell distribution
4. monthly rainfall climatology
5. TMAX distribution
6. TMIN distribution
7. SRAD distribution
8. episode length distribution
```

图中明确区分：

```text
fitting observed
independent observed
synthetic ensemble
```

不制作 PPT。

---

# 十九、结果目录

```text
results/yc_random_weather_episode_validation/004_02/
```

建议：

```text
episodes/
    seed_1001.csv
    ...
    seed_1100.csv

reproducibility/
generation_qc/
validation/
figures/

summary.json
experiment_log.md
```

---

# 二十、关键结果表

至少生成：

```text
episode_generation_qc.csv
fixed_window_weather_by_seed.csv
full_crop_season_weather_by_seed.csv

rainfall_validation_summary.csv
monthly_rainfall_validation.csv
temperature_validation_summary.csv
srad_validation_summary.csv

cross_correlation_validation.csv
serial_correlation_validation.csv

weather_seed_reproducibility.csv
weather_seed_diversity.csv
```

---

# 二十一、Gate 判定

最终分开判断：

```text
generation_status
reproducibility_status
diversity_status
rainfall_status
temperature_status
srad_status
dependence_structure_status
```

然后：

```text
random_weather_episode_quality_status =
PASS
PASS_WITH_NOTES
FAIL
```

### PASS / PASS_WITH_NOTES

如果没有发现会使 PPO 训练输入失真的系统性问题：

```text
yc_random_weather_ready_for_ppo_pilot = YES
```

下一步：

```text
004_03 YC random-weather PPO pilot
```

### FAIL

必须明确失败来自：

```text
generation
rainfall
temperature
SRAD
correlation structure
```

不能笼统写“weather failed”。

---

# 二十二、重新解释全年 WGEN blocker

报告中明确记录：

```text
The absence of a verified automated full-calendar-year WGEN export path
does not block the episode-level random-weather RL workflow used here.
```

即：

```text
full-year climatology validation:
DEFERRED / OPTIONAL ADDITIONAL VALIDATION

episode-level random-weather validation:
ACTIVE REQUIRED GATE
```

---

# 二十三、中文报告

生成：

```text
docs/yc_random_weather_episode_ensemble_validation.md
```

报告结构至少：

```text
1. Why episode-level validation
2. Relation to Wang et al. 2025
3. Frozen WGEN inputs
4. Seed design
5. Fixed-window definition
6. Generation QC
7. Reproducibility
8. Diversity
9. Rainfall validation
10. Temperature validation
11. SRAD validation
12. Dependence structure
13. Crop-season auxiliary statistics
14. Fitting vs independent observed comparison
15. Limitations
16. Final decision
17. PPO readiness
18. Next step
19. Git status
```

---

# 二十四、禁止过度声称

即使通过，也只能表述：

> The WGEN-generated episode-level weather realizations are sufficiently consistent with the observed YC growing-season climate statistics for use in the planned random-weather PPO pilot.

不能写：

```text
WGEN perfectly reproduces YC climate
full annual climate validated
future climate validated
extreme climate validated
```

---

# 二十五、测试

至少报告：

```text
100 episode generation completion
5 same-seed rerun checks
different-seed diversity
fixed-window extraction tests
physical QC
aggregation consistency
no validation leakage
```

---

# 二十六、Git

local commit：

```text
test: validate YC random-weather episode ensemble
```

只提交本任务相关文件。

未经用户明确批准：

```text
git push = NO
```

---

# 二十七、最终终端摘要

必须输出：

```text
=== YC RANDOM-WEATHER EPISODE ENSEMBLE VALIDATION SUMMARY ===

reference_method:
Wang_2025_style_episode_weather: YES

fitting_period:
independent_validation_period:
CNYC_CLI_sha256:

synthetic_episode_count:
weather_seed_range:
fixed_common_window:

generation_status:
episodes_passed:
episodes_failed:

same_seed_reproducibility:
different_seed_diversity:

fitting_rain_mean:
validation_rain_mean:
synthetic_rain_mean:

fitting_rain_range:
validation_rain_range:
synthetic_rain_range:

rainfall_status:
temperature_status:
srad_status:
dependence_structure_status:

systematic_bias_detected:

random_weather_episode_quality_status:
yc_random_weather_ready_for_ppo_pilot:

full_year_wgen_validation:
full_year_wgen_is_ppo_blocker: NO

PPO_run: NO
runtime_modified: NO
CNYC_CLI_modified: NO
WGEN_refit: NO
validation_data_used_for_fitting: NO
ppt_created: NO

recommended_next_step:

report_md:
results_directory:

tests_status:
git_commit:
git_push: NO
github_backup_status:
```

成功时：

```text
random_weather_episode_quality_status: PASS or PASS_WITH_NOTES
yc_random_weather_ready_for_ppo_pilot: YES
recommended_next_step: 004_03 YC random-weather PPO pilot
```
