# 028_06 已筛选年份四基线复用与缺口登记

## 结论

已逐项检查 40 个站点—年份—情景单元；34 个已有快照通过必需文件检查，可直接复用；真正缺失 6 个。
本任务运行 DSSAT 0 季、训练 0 次。

## 真正缺失情景

|站点|年份|情景|输入变体|
|---|---:|---|---|
|YC|2008|official_extension_expert|013_01_forward_screening|
|FQ|2013|official_extension_expert|014_01_weather_year_derived|
|FQ|2014|official_extension_expert|014_01_weather_year_derived|
|FQ|2019|official_extension_expert|014_01_weather_year_derived|
|FQ|2020|official_extension_expert|014_01_weather_year_derived|
|FQ|2023|official_extension_expert|014_01_weather_year_derived|

## 来源边界

- HLA 四个年份来自 020_11 prepared-adapter 链路；这不是未经转换的原始 treatment。
- YC2008 来自 013_01 forward-screening 链路。
- FQ 五个年份来自 014_01 weather-year 派生链路；其中 recorded 为 recorded_shifted。
- official expert 不得由 recorded 或其他年份的结果替代。

## 下一步硬边界

下一任务最多运行 6 季 official expert；其余 34 个情景不得重复运行。
