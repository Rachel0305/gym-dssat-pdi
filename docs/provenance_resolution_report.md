# SYA / LCA 非降水来源缺口追踪报告

范围：`remaining_nonrain_provenance_gaps.csv`的484个站点—日期—变量项；只读追踪，未修改WTH、DSSAT/WGEN/PPO配置，未运行模型或模拟。

## 结论

|站点|项数|已证实外部补测|确认旧均值填补（数值层面）|无法确定最终写入来源|gate|
|---|---:|---:|---:|---:|---|
|SYA|462|0|462|0|BLOCKED_BY_DATA_PROVENANCE|
|LCA|22|0|0|22|BLOCKED_BY_DATA_PROVENANCE|

SYA的462项可确认最终WTH采用了历史“站点×月份”均值（按WTH一位小数精度）；这些值没有逐日外部观测证据。LCA的22项同样等于旧均值，但LCA训练年后来发生天气补充，在缺少逐项导入记录时不能把“数值相同”升级为“最终来源已证实”，故列为无法确定。没有任何一项建立“原始XLS空白→外部下载数据→最终WTH”的逐日变量级证据链。

## 证据链

1. `src/weather_preprocess.py`：`build_specs`从`my_data`指定SYA辐射/温度XLS及LCA的`D32.xls`/`T2.xls`；`normalize_source`识别空白；`merge_weather`按站点日期合并；`reindex_full_years`补齐日期；`fill_missing`对SRAD/TMAX/TMIN按站点跨所有年份同月均值填补，最后输出`weather_clean/{site}_weather_cleaned.csv`与`missing_value_fill_log.csv`。`weather_clean/data_check_report.md`和`docs/experiment_records/weather_data_check_report.md`记录过该流程。审计中以原始缺口日期从cleaned CSV中排除对应变量，重新计算月份均值；484项与cleaned精确一致，最终WTH按一位小数也全部一致。

2. `weather_clean_qc/{site}_weather_cleaned_qc.csv`和`Leave_One_experiments/wth_generated_qc/{site}/{site}{year}.WTH`在484项均与旧均值一致；`Leave_One_experiments/wth_generated_qc/wth_generation_summary_qc.csv`保留生成版天气文件清单。`src/run_weather_qc_030_00.py`的030_00结果记录仅对HLA两处值作校正，没有SYA/LCA的目标项校正。最终WTH与originIC目录相同年份文件SHA256一致，包括LCA的lowIC目录。

3. SYA 2005–2013最终WTH与旧生成版逐日四变量比较只有2006、2011各1个非目标SRAD值差异；最终WTH与`weather_clean/SYA_weather_cleaned.csv`的全年逐值比较在一位小数误差内一致。462个目标项没有变动，因此确认其现有值是旧均值填补值。这里确认的是WTH数值及历史生成链的对应，不声称存在完整的旧命令运行日志。

4. LCA 2005–2013最终WTH与旧生成版在多个年份已有补充差异；22个目标项仍等于旧均值。用户此前确认的历史下载补充解释了差异背景，但项目现存`data/external`目录只见YC、HLA/FQ相关下载子目录；`my_data`中SYA/LCA日值XLS是原始观测输入，未发现独立的这22项补测文件。备份内发现的SYA/LCA WTH为实验输入/渲染副本，不是带原始发布者、下载时间及逐行导入关系的独立补测材料。没有找到足以把这22项判为“已证实补测”的manifest。

## 判定边界

“确认旧均值填补”指最终WTH数值与可复算的历史均值填补输出一致，且SYA整年序列与旧来源链一致；不表示已经有可用于WGEN拟合的独立实测资料。LCA同值22项可疑为旧填补残留，但由于后续补充流程覆盖了同年其他天气值，无法确定最终编辑时对这22项采用的具体来源。数值巧合不能在没有逐项记录时排除。

本次不改变`final_gate.json`：SYA仍因462项旧均值填补缺少独立补测/可接受填补依据而BLOCKED；LCA仍因22项来源无法确定而BLOCKED。若要解除，应提供这些日期和变量对应的补测文件与映射/导入记录，或明确接受这些旧均值作为WGEN拟合输入并独立复审。RAIN空白=0和LCA其余1113处差异的既有修订结论维持。

逐项表：`results/sy_lc_random_weather_015/nonrain_provenance_resolution_484.csv`；机器摘要：`results/sy_lc_random_weather_015/nonrain_provenance_resolution_summary.json`。
