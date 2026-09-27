# 003_06_06：YC 随机天气完整 QC 与 DSSAT 单季 crop-output smoke

## 一、任务定位

当前 YC 随机天气增强项目已完成：

```text
2004–2013 frozen train weather ✅
        ↓
WGEN parameter fitting ✅
        ↓
corrected CNYC.CLI ✅
        ↓
DSSAT 4.8.0.024 reads CLI ✅
        ↓
weather_seed interface ✅
        ↓
same-seed reproducibility ✅
        ↓
different-seed diversity ✅
        ↓
basic physical QC ✅
        ↓
full weather QC + DSSAT crop-output smoke   ← CURRENT TASK
        ↓
random-weather PPO pilot
```

上一轮 `003_06_05_02` 已确认：

```text
formal corrected CLI:
results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI

SHA256:
65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929

runtime:
DSSAT 4.8.0.024

seeds:
101a PASS
101b PASS
102 PASS
103 PASS
104 PASS
105 PASS

same seed reproducibility:
101a == 101b

different seed diversity:
101–105 -> 5 distinct weather sequences

basic physical QC:
PASS

WGEN seed pilot:
WGEN_SEED_PILOT_PASS
```

本任务不再修改 WGEN 参数，也不重新生成 CLI。

---

# 二、本任务核心目标

本轮必须分成两个 gate：

## Gate A：随机天气完整 QC

回答：

> 当前 YC WGEN 生成的随机天气，在固定比较口径下，是否与 2004–2013 train climate 保持合理一致，同时又具有必要的随机差异？

重点检查：

```text
RAIN
SRAD
TMAX
TMIN
wet days
dry spells
extremes
monthly climatology
seasonal distribution
```

## Gate B：DSSAT crop-output smoke

回答：

> 这些随机天气进入 DSSAT 后，玉米模拟是否能稳定运行，并产生合理的 phenology、yield、水分和氮状态？

并特别审查上一轮 runtime warnings 是否影响 crop outputs。

---

# 三、输入保护

唯一正式 corrected CLI：

```text
results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI
```

SHA256：

```text
65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929
```

冻结 train weather：

```text
results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv
```

SHA256：

```text
4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34
```

禁止修改：

```text
corrected CLI
frozen train weather
WGEN fitting formulas
wet-day definition
PPO reward
action space
observation space
other sites
```

---

# 四、禁止使用 validation weather 进行任何拟合或调参

2014–2023 validation weather：

```text
禁止用于：
WGEN fitting
threshold tuning
parameter adjustment
bias correction
acceptance threshold calibration
```

如确有需要，可在报告最后作为未来独立验证阶段提出，但本任务不得使用。

---

# 五、Gate A：必须先解决比较口径问题

上一轮不同 seed 捕获长度不同：

```text
seed 101: 120 days
seed 102: 117 days
seed 103: 117 days
seed 104: 116 days
seed 105: 118 days
```

原因：

> DSSAT 在不同随机天气下作物仿真终止日不同。

因此禁止直接把不同长度序列的累计天气量当成完全可比的气候差异。

本轮必须建立两个独立口径：

## 5.1 Fixed common calendar window

使用所有 seed 都覆盖的共同日期窗口。

例如：

```text
start = 2008-06-01
end = 所有 seed 中最早结束日
```

必须由实际数据自动求：

```text
common_start
common_end
common_day_count
```

在该共同日期窗口上比较：

```text
RAIN
SRAD
TMAX
TMIN
wet days
dry spells
extremes
```

## 5.2 Full simulation-period window

每个 seed 保留自己的完整 simulation period，单独报告：

```text
simulation days
cumulative rain
rainy days
longest dry spell
mean weather variables
```

但明确标注：

```text
NOT DIRECTLY COMPARABLE FOR CUMULATIVE TOTALS
```

避免统计口径混用。

---

# 六、Gate A：训练期 reference 的正确构造

使用 frozen 2004–2013 train weather。

由于随机天气 smoke 当前对应 YC 2008 生长季窗口：

应构造与 common calendar window 对齐的历史 reference，例如：

```text
对 2004–2013 每一年取相同 calendar dates
```

如果存在闰日/日期差异，必须明确处理。

生成：

```text
historical_common_window_reference.csv
```

对每一年单独计算：

```text
RAIN total
wet days
max daily rain
longest dry spell
mean TMAX
mean TMIN
mean SRAD
min TMIN
max TMAX
max SRAD
```

再汇总：

```text
mean
sd
median
min
max
P05
P25
P75
P95
```

注意：

> 这是 train climate descriptive reference，不是 validation，也不是 acceptance threshold tuning。

---

# 七、Gate A：月尺度 climatology QC

如果 common window 跨多个自然月：

对实际覆盖月份计算：

```text
monthly RAIN
monthly wet days
monthly mean TMAX
monthly mean TMIN
monthly mean SRAD
monthly max daily RAIN
```

对比：

```text
WGEN seed 101–105
vs
2004–2013 train same-month climatology
```

输出：

```text
monthly_weather_qc.csv
```

不要因为单个 seed 偏离历史均值就自动判异常。

重点观察：

```text
是否落在历史可接受分布附近
是否出现系统性偏移
是否所有 seed 都朝同一方向偏
```

---

# 八、Gate A：降雨结构 QC

重点分析：

```text
wet-day frequency
dry-day frequency
dry spell length
wet spell length
daily rainfall distribution
heavy rainfall tail
```

至少输出：

```text
rainfall_structure_qc.csv
```

包括：

```text
seed
common_window_days
wet_days
wet_day_fraction
rain_total
mean_rain_on_wet_day
median_rain_on_wet_day
P90_rain
P95_rain
max_daily_rain
longest_dry_spell
longest_wet_spell
```

与 2004–2013 train same-window reference 对比。

---

# 九、Gate A：温度 QC

分别对：

```text
TMAX
TMIN
```

分析：

```text
mean
sd
P05
P50
P95
min
max
```

并检查：

```text
TMAX >= TMIN
```

继续做 descriptive extreme counts：

```text
hot days
cold nights
```

阈值优先使用：

```text
train weather empirical percentile
```

例如：

```text
historical train P90/P95
```

而不是人为拍固定阈值。

必须明确：

> 这些 percentile 只用于描述 generated weather 与 train climate 的相对位置，不用于重新拟合 WGEN。

---

# 十、Gate A：SRAD QC

分析：

```text
mean
sd
P05
P50
P95
min
max
```

检查：

```text
SRAD >= 0
```

比较：

```text
seed distributions
vs
train common-window reference
```

重点检查是否出现：

```text
systematically too low
systematically too high
unphysical extreme
almost no seed variability
```

---

# 十一、Gate A：随机天气总体 QC 判定

不要要求每个 seed 都“像历史平均年”。

随机天气本来就应有 variability。

本轮判定重点是：

```text
1. physical validity
2. no obvious systematic bias
3. generated values overlap historical train distribution reasonably
4. rainfall occurrence/amount not degenerate
5. seeds are diverse
6. no impossible extremes
```

状态建议：

```text
PASS
PASS_WITH_NOTES
FAIL_SYSTEMATIC_BIAS
FAIL_PHYSICAL
FAIL_DEGENERATE_VARIABILITY
```

不得因为某个 seed 比历史均值干/湿就直接判失败。

---

# 十二、Gate A 图表

生成用户可检查的图表，至少：

```text
1. common-window cumulative RAIN by seed + historical years
2. wet-day counts
3. longest dry spell
4. TMAX distribution
5. TMIN distribution
6. SRAD distribution
7. monthly RAIN comparison
```

图表应：

```text
一个图一张
标题清楚
单位完整
seed 与 historical reference 区分明确
不使用 validation data
```

保存到：

```text
results/yc_wgen_cli_pilot/003_06_06/weather_qc/figures/
```

---

# 十三、Gate B：审查 runtime warnings

上一轮六次均存在：

```text
PHOTO method L -> C
FileX latitude missing
FileX longitude missing
FileX elevation missing
CYCRDin / CXCRDin / CELEVin not passed
soil STONES default
soil ADCOEF default
cultivar-related field warnings
```

本轮先逐项分类：

```text
warning
source
expected value
actual runtime value
likely affected process
fatal/nonfatal
crop-output relevance
needs_fix_before_smoke
needs_fix_before_PPO
```

生成：

```text
runtime_warning_audit.csv
```

---

# 十四、FileX latitude / longitude / elevation

必须明确：

```text
YC known station metadata:
LAT = 36.830
LONG = 116.570
ELEV = 22 m
```

审查：

```text
FileX 是否本来应该提供这些值
Gym-DSSAT wrapper 是否在传递时丢失
DSSAT runtime 是否另从 weather station 获取
warning 是否导致实际 runtime 使用 0
哪些模块读取这些值
```

不得自动把 CLI 中的 station metadata 等同于 FileX metadata。

如果 crop simulation 所需：

在 isolated copy 上做最小修复。

禁止修改历史 source FileX。

---

# 十五、STONES / ADCOEF 默认值

确认：

```text
STONES
ADCOEF
```

分别：

```text
含义是什么
DSSAT 默认值是什么
当前 SOIL.SOL 是否缺失
当前 YC 历史 baseline 是否也一直使用默认值
是否影响 water balance / N / root / crop output
```

关键判断：

> 如果 historical baseline 与当前 random-weather smoke 都使用同样默认值，则它可能不是“随机天气新引入的问题”，但仍需记录其模型不确定性。

不要在没有来源数据时擅自填一个值。

---

# 十六、PHOTO method warning

确认：

```text
PHOTO method L -> C
```

具体表示什么。

检查：

```text
为什么发生
是否为 DSSAT 正常 fallback
是否改变 maize photosynthesis logic
historical YC baseline 是否同样出现
```

如果只是既有 baseline 行为：

记录为：

```text
BASELINE_EXISTING_WARNING
```

不要为了消除 warning 随意改模型设置。

---

# 十七、cultivar warning

逐条确认 cultivar warning：

```text
field name
source cultivar file
runtime expected field
current value
fallback/default
historical baseline comparison
```

重点判断：

> 是否会影响 phenology / yield。

如果是既有且 historical YC calibration 就使用的输入：

不要擅自改 cultivar 参数。

---

# 十八、Gate B：建立 historical-weather control smoke

为了隔离“random weather”本身的影响：

必须运行一个：

```text
historical-weather control
```

要求：

```text
same YC treatment
same FileX
same cultivar
same soil
same management
same runtime
random_weather=False
historical 2008 weather
```

这个 control 用来回答：

> runtime warning 是否在 historical baseline 中也存在？

并提供 crop-output 对照。

---

# 十九、Gate B：random-weather crop smoke

至少选择：

```text
weather_seed = 101
weather_seed = 104
```

理由：

```text
101 relatively wet
104 relatively dry
```

但不得把它们称为正式 best/worst scenario。

如运行成本低，也可运行：

```text
101–105
```

但最低要求是两个 seed。

---

# 二十、Gate B：crop output 指标

至少提取：

```text
planting date
anthesis date
maturity / termination date
season length
final grain yield / grnwt
biomass / topwt
LAI maximum
cumulative irrigation
cumulative fertilizer
soil water stress indicators if available
nitrogen stress indicators if available
N uptake / soil N outputs if available
```

不要为了凑指标重新修改 Gym observation。

只读取当前 runtime / DSSAT 已有输出。

---

# 二十一、phenology 检查

比较：

```text
historical-weather control
seed 101
seed 104
```

关注：

```text
anthesis timing
maturity timing
season length
```

随机天气导致 phenology 有变化是允许的。

失败条件包括：

```text
no emergence without plausible reason
maturity impossible
simulation terminates immediately
phenology dates missing unexpectedly
grossly impossible stage order
```

---

# 二十二、yield / biomass smoke

本轮不是做 performance benchmark。

只检查：

```text
yield finite
yield >= 0
biomass finite
biomass >= yield
no obvious numerical explosion
```

并描述：

```text
historical control
vs
seed 101
vs
seed 104
```

不要因为 random weather yield 低于 historical 2008 就判失败。

---

# 二十三、水分与氮状态 smoke

如果 DSSAT outputs 可获得：

检查：

```text
soil water
water stress
N stress
N uptake
fertilizer events
irrigation events
```

重点：

```text
finite
range plausible
events preserved
no missing due wrapper failure
```

本轮不评价 PPO 管理策略。

---

# 二十四、warnings 的最终分类

所有 warnings 最终必须放到三类：

```text
A. harmless / informational for current purpose
B. existing baseline limitation
C. must fix before PPO
```

如果有 C 类：

```text
DSSAT_CROP_SMOKE_READY = NO
```

并明确最小修复动作。

---

# 二十五、湿日阈值敏感性仍然不在本轮处理

当前 CLI 继续：

```text
RAIN > 0.0 mm
```

本轮不比较：

```text
RAIN >= 0.254 mm
```

理由：

> 先完成主链 crop-output smoke；湿日阈值敏感性作为独立后续 robustness experiment。

不得重新生成 alternative CLI。

---

# 二十六、成功 Gate

本任务通过需要：

## Gate A

```text
weather_qc_status = PASS or PASS_WITH_NOTES
```

且：

```text
no physical invalidity
no obvious systematic generation failure
```

## Gate B

```text
runtime warnings classified
historical control runs
random-weather crop smoke runs
phenology/yield/biomass finite and interpretable
no fatal crop-model issue
```

最终：

```text
YC_RANDOM_WEATHER_DSSAT_SMOKE_PASS
```

只有达到该状态，下一阶段才允许开始：

```text
random-weather PPO pilot
```

---

# 二十七、本轮禁止事项

禁止：

```text
formal PPO training
PPO hyperparameter search
reward redesign
action space changes
observation changes
WGEN refit
wet-day threshold change
use validation weather for fitting
modify other sites
manual tuning to improve yield
```

---

# 二十八、结果目录

保存到：

```text
results/yc_wgen_cli_pilot/003_06_06/
```

建议：

```text
weather_qc/
├── fixed_window_weather.csv
├── historical_common_window_reference.csv
├── monthly_weather_qc.csv
├── rainfall_structure_qc.csv
├── temperature_qc.csv
├── srad_qc.csv
├── weather_qc_summary.json
└── figures/

runtime_warning_audit/
├── runtime_warning_audit.csv
├── warning_evidence/
└── warning_summary.json

crop_smoke/
├── historical_control/
├── seed_101/
├── seed_104/
├── crop_output_summary.csv
├── phenology_summary.csv
└── crop_smoke_summary.json

experiment_log.md
```

---

# 二十九、中文 Markdown 报告

生成：

```text
docs/yc_wgen_weather_qc_and_dssat_smoke.md
```

至少包括：

```text
1. task scope
2. inputs and provenance
3. fixed comparison window
4. historical train reference
5. rainfall QC
6. temperature QC
7. SRAD QC
8. weather QC decision
9. runtime warning audit
10. historical-weather control
11. random-weather crop smoke
12. phenology
13. yield/biomass
14. water/N status
15. warning classification
16. readiness for PPO
17. remaining issues
18. next step
19. files changed
20. git status
```

正文使用中文。

---

# 三十、PPT 任务记录

生成：

```text
docs/yc_wgen_weather_qc_and_dssat_smoke.pptx
```

建议 8–10 页：

```text
1. 项目位置
2. fixed-window QC 设计
3. rainfall
4. TMAX/TMIN
5. SRAD
6. runtime warning audit
7. historical control vs random weather
8. crop phenology/yield
9. water/N smoke
10. PPO readiness
```

---

# 三十一、Codex prompt 保存

保存：

```text
prompts/003_06_06_yc_wgen_weather_qc_and_dssat_smoke.md
```

---

# 三十二、Git 与 GitHub

完成后：

1. `git diff`
2. 只 stage 本任务文件
3. local commit
4. commit message：

```text
test: validate YC WGEN weather and DSSAT crop smoke
```

5. 不清理无关工作区
6. 未经用户明确批准：

```text
git push = NO
```

记录：

```text
GitHub backup pending explicit user approval
```

---

# 三十三、最终终端摘要

必须直接返回：

```text
=== YC WGEN WEATHER QC + DSSAT CROP SMOKE SUMMARY ===

corrected_cli:
corrected_cli_sha256:
cli_hash_verified:

frozen_train_weather:
train_weather_sha256:
validation_weather_used_for_fitting: NO

common_window_start:
common_window_end:
common_window_days:

weather_qc_status:
rainfall_qc:
temperature_qc:
srad_qc:
systematic_bias_detected:

historical_control_status:

runtime_warning_total:
warning_class_A:
warning_class_B:
warning_class_C:
warnings_blocking_ppo:

random_weather_crop_seeds_run:

phenology_status:
yield_biomass_status:
water_status:
nitrogen_status:

crop_smoke_status:

wet_day_definition_changed: NO
wgen_refit: NO
ppo_training_run: NO
other_sites_modified: NO

yc_random_weather_dssat_smoke_status:

recommended_next_step:

report_md:
report_pptx:
results_directory:

tests_status:
git_commit:
git_push: NO
github_backup_status:
```

成功时：

```text
weather_qc_status: PASS or PASS_WITH_NOTES
crop_smoke_status: PASS
warnings_blocking_ppo: NO
yc_random_weather_dssat_smoke_status: YC_RANDOM_WEATHER_DSSAT_SMOKE_PASS
recommended_next_step: design YC random-weather PPO pilot
```

如果存在阻塞 warning：

```text
warnings_blocking_ppo: YES
yc_random_weather_dssat_smoke_status: BLOCKED_BEFORE_PPO
```

必须具体说明是哪一个 warning 及为什么。

---

# 三十四、完成标准

本任务只有在以下全部完成后结束：

1. corrected CLI hash 已核验；
2. frozen train weather hash 已核验；
3. fixed common calendar window 已建立；
4. 2004–2013 same-window historical reference 已建立；
5. rainfall QC 完成；
6. TMAX/TMIN QC 完成；
7. SRAD QC 完成；
8. 不混用 full-period 与 fixed-window 累计量；
9. runtime warnings 全部逐项审计；
10. historical-weather control smoke 完成；
11. 至少两个 random-weather crop smoke 完成；
12. phenology/yield/biomass 检查完成；
13. water/N 状态尽可能检查；
14. warning 分成 A/B/C 三类；
15. 明确是否存在 PPO blocker；
16. 不修改 WGEN fitting；
17. 不改变 wet-day definition；
18. 不使用 validation fitting；
19. 不运行正式 PPO；
20. 不修改其他站点；
21. 生成中文 Markdown + PPT；
22. 保存完整实验记录；
23. local git commit；
24. 未经用户批准不执行 `git push`。
