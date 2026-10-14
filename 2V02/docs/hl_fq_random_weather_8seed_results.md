# HL / FQ 随机天气 8-seed PPO：阶段结果

**总判定：** `BLOCKED_BY_DATA_PROVENANCE`  
**停止位置：** `STOPPED_AT_WEATHER_QC`（WGEN 之前）  
**日期：** 2026-10-04（Asia/Shanghai）

## 结果摘要

| 项目 | HLA | FQA |
|---|---|---|
| 历史天气训练期 | 2004–2013 | 2005–2013 |
| 原始 RAIN 空白 | 2,691 / 3,653 天 | 2,892 / 3,287 天 |
| 可筛选官方站候选填补 | 2,580 天 | 2,735 天 |
| 仍无可核实 RAIN 来源 | 111 天，均在 2013 | 157 天，均在 2013 |
| 随机天气 PPO seeds | 0 / 8 | 0 / 8 |
| 历史 vs random 对照 | 未生成 | 未生成 |

2013 年雨量未通过外部站点覆盖后，没有完整可信的 fitting weather。附件要求在关键变量不可可靠补齐时停止，因此没有为了达到训练配额而把未知降水改成零。HLA/FQA 的 WGEN 随机天气与 8-seed PPO 结果目前均为空；这里没有任何 yield、灌溉、N、`PFP_N`、reward 或年度策略比较结论。

## 未启动与未修改项

- HLA/FQA fitting weather CSV、站点 `.CLI`、100-member weather pool：未生成。
- climate coverage、DSSAT runtime smoke、2014–2023 replay：未运行。
- HLA/FQA PPO seeds 0–7：全部未启动。
- historical baseline、reward、16-action space、mask、安全约束、IC、100K checkpoint 规则和验证年：未修改。
- 原始工作簿、历史 `.WTH`、已有模型和评估结果：未改写。

## 可复核证据

- 数据源、缺失与 NOAA/CMA 补源审计：[hl_fq_wgen_weather_source_audit.md](hl_fq_wgen_weather_source_audit.md)
- WGEN 与天气 QC 状态：[hl_fq_wgen_weather_generation_qc.md](hl_fq_wgen_weather_generation_qc.md)
- 最终 machine-readable gate：[weather_source_audit_final.json](../results/hl_fq_random_weather_8seed/source_audit/weather_source_audit_final.json)
- 原始候选、站点记录、日期级 gapfill 仅保存在 `results/hl_fq_random_weather_8seed/source_audit/`；候选填补行均未晋升为 fitting weather。

## 继续实验所需输入

请将 HLA 与 FQA 2013 年缺失雨量日期的可追溯官方逐日数据放到项目目录，或提供能够核实相应空白值语义的本地数据字典/站点记录。收到数据后先重新运行 source audit 和完整拟合期物理 QC，再决定是否进入 WGEN。中国气象数据网目录有国家级日值气象数据产品，但需实名账户访问；本轮没有该原始数据文件，也没有操作任何账户。

## Git 状态

这是一个已归档的审计阶段，后续 WGEN/PPO 尚未完成。未执行 GitHub push；项目 `AGENTS.md` 要求 push 前获得用户确认。是否提交本阶段只包含本任务的 audit/report artifacts，以最终工作区提交动作回报为准。
