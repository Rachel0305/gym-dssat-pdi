# 028_08 HLA2010 阶段型 MaskablePPO 固定权重跨年迁移

状态：`completed`

本任务复用三个 HLA2010 selected checkpoint，在 HLA2007/2015/2016/2022 上零训练确定性评估；四基线全部复用，未重跑。

## 导师规则结果

|year|winner seeds|yield wins|WP_ET wins|PFP_N wins|2/3通过|
|---:|---:|---:|---:|---:|---|
|2007|2/3|1|0|1|True|
|2015|2/3|2|1|2|True|
|2016|2/3|2|0|0|True|
|2022|1/3|0|0|1|False|

其余两个未领先指标只在 CSV 中报告差值，不设置未经导师确认的接近阈值。

训练步数：0。
