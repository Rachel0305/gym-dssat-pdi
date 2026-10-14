# YC WGEN SRAD低尾与联合天气结构诊断（004_20）

## 结论摘要

```text
srad_gap_character = MIXED_DEFICIT
fitting_low_tail_status = FITTING_COVERS_OBSERVED_LOW_TAIL
generated_vs_fitting_status = MIXED
joint_structure_status = MULTIVARIATE_GAP
next_weather_design_action = DESIGN_MULTIVARIATE_CORRECTION
```

EXACT78是主分析；FULL80只作包含provisional 1003/1036的敏感性；held-out 20保持隔离。所有Observed年份只用于天气域诊断，没有用于拟合任何修正参数。

## 数据完整性与单位

fitting CSV SHA256 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`，冻结CLI SHA256 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`；两者与004_01/003_06_05_02记录一致。100个canonical CSV按004_17 verified manifest逐文件校验。Observed daily weather复用004_18A `observed_rows()`，由官方WTH、FileX treatment-1 PDATE和seed5 evaluation trace最大DAP确定日序列。

SRAD字段没有进行×10或÷10换算。CLI参数定义文件记载SRAD单位为MJ m⁻² d⁻¹；Observed来自DSSAT WTH的SRAD列，runtime archive来自PDI daily-state SRAD字段。QA逐组报告缺失、-99、0和负值；`-99`仅按缺失sentinel识别并报告，零值保留，其他负数不静默删除。

| 数据组 | unit数 | SRAD最小值 | SRAD最大值 | unit均值均值 |
|---|---:|---:|---:|---:|
| EXACT78 | 78 | 4.44 | 33.43 | 16.45 |
| FITTING | 11 | 0.00 | 37.06 | 13.52 |
| HELDOUT20 | 20 | 4.60 | 33.39 | 16.08 |
| OBSERVED | 10 | 1.50 | 30.70 | 17.94 |
| PROVISIONAL2 | 2 | 5.19 | 29.62 | 16.80 |

fitting全期P1/P5阈值分别为 **0.766 / 2.468 MJ m⁻² d⁻¹**。低辐射天气类型的TMAX分界使用fitting作物日历月份TMAX中位数 **29.50°C**；均未用Observed估阈值。

## 是否仅为单日最小值问题

| 指标 | EXACT78 | FULL80 | held-out20 | 越界方向 |
|---|---:|---:|---:|---|
| season SRAD min | 10/10 | 10/10 | 10/10 | Observed低于pool下界 |
| season SRAD P1 | 8/10 | 8/10 | 8/10 | Observed低于pool下界 |
| season SRAD P5 | 3/10 | 3/10 | 3/10 | Observed低于pool下界 |
| SRAD<5日数 | 2/10 | 2/10 | 8/10 | Observed高于pool上界 |
| 低于fitting P5日数 | 4/10 | 4/10 | 4/10 | Observed高于pool上界 |
| 最小3日均值 | 2/10 | 2/10 | 3/10 | Observed低于pool下界 |
| 最小5日均值 | 0/10 | 0/10 | 1/10 | Observed低于pool下界 |

判定遵循预设规则：除min外，至少两个非min指标在多数Observed年份（≥6/10）低于EXACT78支持，且held-out有同方向缺口，才认定持续低尾。EXACT78越界指标：srad_p1=8/10, srad_p5=3/10, low_days_lt5=2/10, low_days_below_fit_p5=4/10, min_3day_mean=2/10, min_5day_mean=0/10；FULL80：srad_p1=8/10, srad_p5=3/10, low_days_lt5=2/10, low_days_below_fit_p5=4/10, min_3day_mean=2/10, min_5day_mean=0/10；held-out：srad_p1=8/10, srad_p5=3/10, low_days_lt5=8/10, low_days_below_fit_p5=4/10, min_3day_mean=3/10, min_5day_mean=1/10。达到多数标准的非min指标：srad_p1。其中只有SRAD P1一项达到标准，未满足“两项”持续低尾门槛；但它与10/10越界的min及联合结构信号并存，因此归为MIXED，而不是称为仅单日min或无缺口。

因此本轮分类为 `MIXED_DEFICIT`。这比单看004_19的srad_min更严格；同时提供P0.5/P1/P2.5/P5/P10、绝对阈值日数、fitting P1/P5阈值日数、连续spell及rolling 3/5/7日均值。逐unit与逐DAP阶段数据见CSV。

## fitting basis 与 WGEN

fitting 2004–2013 pooled daily SRAD最低值为 **0.00**。同月对照中，Observed月度P1低于fitting年度同月P1支持的比例为 `0.0%`，P5比例为 `2.5%`（仅在可配对月份上计算）。这些是同月覆盖描述，不是独立样本显著性检验。

EXACT78 crop-calendar月匹配日值的Spearman(SRAD,RAIN)为 `-0.456`、Spearman(SRAD,TMAX)为 `0.438`，fitting对应为 `-0.392` / `0.668`。以全期fitting P5阈值 `2.468` 定义低SRAD时，EXACT78没有低阈值日，故wet-day占比不可估；fitting为 `83.7%`，Observed有4个低阈值日且均为wet day。联合差异阈值与效应数值见`decision.json`、`srad_rain_joint_summary.csv`及`srad_temperature_joint_summary.csv`。

当前CLI把SRAD分成按历月的dry/wet条件均值与样本标准差（SDMN/SDSD、SWMN/SWSD），而非显式lower-bound参数。项目证据未证明活动runtime的截断/clip细节或实际是否保留完整低尾；参数估计器与WGEN日生成器是两段不同机制。精确source-to-runtime证据与UNKNOWN项见`wgen_srad_parameter_chain.md`。

## 同月、DAP阶段与复合事件

月匹配月份为 `[6, 7, 8, 9, 10]`。runtime canonical DATE可能来自固定参考日期，因此生成天气月份严格用manifest `season_start + DAP`；Observed用官方WTH真实DATE。fitting没有可信DAP，本报告不为其伪造DAP，只做整年及日历月比较。

| 数据组 | DAP阶段 | median P1 | median P5 | median最小3日均值 |
|---|---|---:|---:|---:|
| EXACT78 | DAP0_30 | 7.02 | 8.56 | 11.33 |
| EXACT78 | DAP31_60 | 6.33 | 7.25 | 9.52 |
| EXACT78 | DAP61_90 | 5.47 | 5.97 | 8.02 |
| EXACT78 | DAP_gt90 | 5.70 | 7.01 | 10.30 |
| FULL80 | DAP0_30 | 7.00 | 8.53 | 11.33 |
| FULL80 | DAP31_60 | 6.33 | 7.17 | 9.52 |
| FULL80 | DAP61_90 | 5.48 | 6.07 | 8.07 |
| FULL80 | DAP_gt90 | 5.70 | 7.01 | 10.30 |
| HELDOUT20 | DAP0_30 | 6.62 | 7.94 | 10.88 |
| HELDOUT20 | DAP31_60 | 6.26 | 6.56 | 9.87 |
| HELDOUT20 | DAP61_90 | 5.26 | 5.71 | 7.30 |
| HELDOUT20 | DAP_gt90 | 5.24 | 6.46 | 9.97 |
| OBSERVED | DAP0_30 | 4.75 | 8.07 | 10.63 |
| OBSERVED | DAP31_60 | 4.18 | 7.10 | 7.97 |
| OBSERVED | DAP61_90 | 4.02 | 5.69 | 7.40 |
| OBSERVED | DAP_gt90 | 8.41 | 9.27 | 12.08 |

低SRAD天气类型均以fitting全期P5和fitting crop-month TMAX中位数划分，且事件类型相互重叠：

| 数据组 | 类型 | 事件数 | 占crop-calendar日比例 |
|---|---|---:|---:|
| FITTING | LOW_SRAD_WET | 36 | 2.35% |
| FITTING | LOW_SRAD_DRY | 7 | 0.46% |
| FITTING | LOW_SRAD_COOL | 43 | 2.81% |
| FITTING | LOW_SRAD_HOT | 0 | 0.00% |
| EXACT78 | LOW_SRAD_WET | 0 | 0.00% |
| EXACT78 | LOW_SRAD_DRY | 0 | 0.00% |
| EXACT78 | LOW_SRAD_COOL | 0 | 0.00% |
| EXACT78 | LOW_SRAD_HOT | 0 | 0.00% |
| FULL80 | LOW_SRAD_WET | 0 | 0.00% |
| FULL80 | LOW_SRAD_DRY | 0 | 0.00% |
| FULL80 | LOW_SRAD_COOL | 0 | 0.00% |
| FULL80 | LOW_SRAD_HOT | 0 | 0.00% |
| HELDOUT20 | LOW_SRAD_WET | 0 | 0.00% |
| HELDOUT20 | LOW_SRAD_DRY | 0 | 0.00% |
| HELDOUT20 | LOW_SRAD_COOL | 0 | 0.00% |
| HELDOUT20 | LOW_SRAD_HOT | 0 | 0.00% |
| OBSERVED | LOW_SRAD_WET | 4 | 0.41% |
| OBSERVED | LOW_SRAD_DRY | 0 | 0.00% |
| OBSERVED | LOW_SRAD_COOL | 4 | 0.41% |
| OBSERVED | LOW_SRAD_HOT | 0 | 0.00% |

## 降雨辅助诊断

fitting天气是完整日历年，但项目没有可核验的2004–2013作物收获DAP终止序列。本报告不把历月总降雨伪装成作物季总降雨；因此fitting seasonal rainfall total标记不可比。辅助CSV提供crop-calendar月份匹配的逐日日雨量低尾以及各WGEN/Observed已冻结季节降雨量。004_19的2014/2016/2019低季节雨量仍对provisional 1036敏感。

## 参数链、边界及行动

`generated_vs_fitting_status = MIXED`，`joint_structure_status = MULTIVARIATE_GAP`。描述性联合筛查阈值预先固定为Spearman差≥0.20、低SRAD wet share差≥15个百分点，或至少2个有≥30日样本的RAIN bin中SRAD P5相对差≥20%；不代表统计显著性。

建议下一步为 **`DESIGN_MULTIVARIATE_CORRECTION`**。随机扩到160目前没有充分直接依据：当前训练和held-out都未覆盖的是当前生成域的问题线索，但更多抽样仍可能偶然产生尾部，不把有限20套held-out当成数学上的不可能性证明。建议先针对本报告锁定的marginal/dependence/fitting问题设计weather-only候选并验证气候支持；在此之前不启动PPO retraining。

## 限制

统计为描述性诊断。低SRAD事件与降雨、温度的共现不能证明因果。fitting全期与作物季窗口并非天然等价；全期threshold用于任务指定的固定诊断口径，同月和DAP窗口另行呈现。provisional seeds 1003/1036不提升为历史精确证据。没有运行WGEN、DSSAT、PPO；没有修改任何既有天气归档或004_19结果。

## 输出索引

`source_data_qa.csv`、`lowest_srad_event_inventory.csv`、`srad_tail_metrics_by_realization.csv`、`srad_tail_metrics_by_stage.csv`、`fitting_vs_generated_srad_tail.csv`、`monthly_srad_tail_comparison.csv`、`srad_rain_joint_summary.csv`、`srad_rain_daily_joint.csv`、`srad_temperature_joint_summary.csv`、`low_srad_weather_type_frequency.csv`、`secondary_rainfall_tail_check.csv`、`fitting_month_matched_observed_low_tail.csv`、`decision.json`、`figures/`。
