# 004_18A YC weather summary for reporting

本任务只读取 004_17 的 `weather_archive_manifest_verified.csv` 所引用的 100 套 canonical daily weather，以及 004_05 seed5 observed evaluation 归档的原始 WTH、FileX 播期和 episode trace 最大 DAP；不训练、不运行 WGEN/DSSAT、不修改旧结果。004_05 trace 的天气四变量均空，不能作为逐日天气来源。78 套 training 和 20 套 held-out 为 historical exact，seed1003/1036 为 provisional。FULL80 与 EXACT78 分开统计。

统一作物季窗口：按 DATE 顺序保留**最后一条 DAP=0**及后续 DAP>0 记录，排除 runtime 启动期 DAP=0；裁剪前后行数须逐套记录。DATE 是 runtime 参考日期，不充当 crop year；crop-year 使用 manifest。Observed 亦按相同规则处理。

RAIN wet-day 定义 `RAIN>0`，dry day 为 `RAIN<=0`；Rx5 为至多 5 日连续滚动降雨最大值（少于5日则不可用）；hot spell 定义连续 `TMAX>32°C`。按 DAP 0–30、31–60、61–90、>90 分段，阶段边界不跨段计算 spell/Rx5。所有统计为描述性，不给正式 GOOD/PARTIAL/POOR verdict。

预先固定 sensitivity 判据：任一 selected metric 的 observed empirical percentile 变化 >=10 个百分点，或 FULL80/EXACT78 的 P5/P95 尾部标记变化，或 min/max support 标记变化，即记录为实质变化；同时呈报每项统计差异，不以这项阈值掩盖差异。seed5 的2014/2019只作天气形态描述，不据此推断产量因果。

产出九张白底浅网格汇报图、realization/stage/sensitivity/observed/presentation CSV，以及两份中文 Markdown。先核验产物完整性再进入实验链归档；该阶段不启动后续 PPO 或天气扩池。
