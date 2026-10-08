# 043 YC 运行对照与 FQA 小规模随机天气归档 Pilot

## 目标与边界

沿 YC 已完成随机天气实验的运行路径，先做一次 YC 2007 年实际模板的只读对照 smoke，确认其原生 `FIELD` 坐标与警告；再做 FQA 2007 年一个处理、预先固定 weather seeds `1001–1005` 的隔离小样本 Pilot，并把 seed `1001` 在新进程重复一次。仅使用零动作捕获实际天气，不运行 PPO、正式 80/20 天气池、8 seeds 或完整气候结论。峰值 RSS 768 MB、每次最多 240 日/120 秒，串行执行。

## 冻结输入与方法

- YC 对照使用已完成 `004_03` 中 `RANDOM_WEATHER_WGEN_ppo_seed_0_2007/YCA_2007_rseed1_1066.jinja2`、其正式 `CNYC.CLI` 及 YC 同包品种/土壤，处理 1、weather seed 1066。只核对运行方式和原生坐标，不把 YC 天气与 FQA 数值比较。
- FQA 使用 041 `CNFQ.CLI`、042 隔离 `WTHER=W` 模板和同一 FQ 品种/土壤。每次新进程将固定 seed 写入 PDI `rseed1_`，由实际 daily state 捕获天气；严禁只从 CLI 或 seed 推算天气。
- 所有生成 realization 的完整逐日 `DATE,DOY,RAIN,SRAD,TMAX,TMIN` 必须先落盘，附规范化字节哈希和输入、年份、处理、seed、进程/episode 上下文。缺任何一条、哈希不匹配或未保存失败运行证据时停止扩展。
- 原始文件、YC 历史结果、FQ 041/042 产物均只读。只向 `results/fqa_archived_weather_pilot_043/` 与新报告写入。

## 判定

1. YC 对照：原生 INP FIELD 经纬高程、IPFLD 警告、CLI/WTHER/rseed1 实际使用、正常结束和天气归档。若 YC 本次运行失败，如实记录，不能由 FileX 的 `-99` 直接推断原生结果。
2. FQA：六次实际天气归档、完整字段、物理基本筛查、seed 1001 重复的逐日哈希、不同 seed 的哈希多样性；年度/单季雨量、湿日、极值和温度辐射仅作描述性小样本检查，不凭五条序列宣布气候分布通过。
3. 比较 YC/FQA 原生 FIELD 表现时区分相同输入占位、相同运行时缺失与已量化功能影响；相同警告只能说明共同限制，不能证明空间计算正确。
4. 输出机器可读 gate 与逐 realization manifest，并形成供后续 FQA 训练入口复用的**逐 episode 实际天气归档合同**：训练开始前先实现捕获，每个 episode 完成即写完整日序列及哈希；同 seed 若在不同上下文天气不同，保留所有实际 realization。仅在归档链可用、天气 QC 与坐标处理边界明确后才考虑 PPO smoke。
