# HLA / FQA 2013 ISD-Lite 降水来源审计（021）

## 结论

现有压缩包确属 NOAA/NCDC ISD-Lite 中国区域 2013 年包，不是 CMA 日值资料。HLA 同站 `507560-99999-2013` 存在，身份可与 014 的 full-ISD `507560-99999`、WMO `50756` 和 GHCN `CHM00050756` 闭合。按完整、不重叠且唯一总量的本地日窗口规则，HLA 2013 有效日数为 **0/365**，014 留下的 **111 天恢复 0 天，仍 unresolved 111 天**。相较 014 的 full-ISD 路线，ISD-Lite 没有更容易形成合法日雨量：014 与本次都没有完整本地日；ISD-Lite 的 1h 降水字段全缺，6h 字段只提供零散窗口，且不含 full-ISD 的其他降水累计记录。它没有改变 HLA 降水来源门禁，也不具备恢复 010 的条件。

FQA 在 ISD history/2013 实际文件中**未找到封丘 exact-name 站**。根据现有 `CNFQ0701.WTH` 天气头坐标（35.020, 114.540, 68 m），最近站为 **570830-99999 / XINZHENG**，距 FQA **84.66 km**。候选前五：570830-99999 XINZHENG (84.7 km, 0 valid days); 549090-99999 DINGTAO (89.2 km, 0 valid days); 538980-99999 ANYANG (120.3 km, 0 valid days); 571930-99999 XIHUA (137.6 km, 0 valid days); 537981-99999 HANDAN/HEIBEI (168.5 km, 0 valid days)。本审计将 50 km 作为直接日值 fallback 的保守空间筛选线；远站只作区域参考。当前正式缺口清单为前序 008/009 机读结果中的 157 天，ISD-Lite 通过已接受站恢复 **0 天**，参考站上可用但不能采纳的天数 **0 天**，剩余缺口 **157 天**。

FQA 2013 D222 有明确数值的 35 天合计 **1144.4 mm**，另有 330 天空白；这只是已记录日的 subtotal，空白没有当 0。NASA POWER 提供完整 365 天，总量 450.56 mm。所选 GHCN `CHM00057083` 只有 106 个有效日，记录值 subtotal 为 399.5 mm，因此它不是完整年总量；它与 D222 明确数值日有 28 天同日数值 overlap。最近 ISD-Lite 站对 D222 明确日的完整日值 overlap 为 0 天。六个 D222 ≥50 mm 事件中，最近站 ±1 日内有 6 个出现正子日观测；最大事件 2013-05-26 的 D222 为 204.0 mm，而最近站同日最大 6h 量为 25.0 mm，NASA 为 59.40 mm、GHCN 为 39.6 mm。最近站距离约 84.7 km，只能提示区域降水事件，不能代表 FQA 点位或验证该日量级。FQA 2013 sanity 继续为 **SUSPICIOUS**；2011–2012 provenance 仍不在该 2013 包覆盖范围内。

## 逐站状态

### HLA

- `HLA_ISDLITE_STATION_IDENTITY = PASS`；2013 同站文件存在，实际记录 2914 条。历史目录位置与 014 中 2013/ GHCN 位置差异已在 identity JSON 中保留，不用静态位置覆盖该差异。
- ISD-Lite 年内 daily 状态：`{"INCOMPLETE_COVERAGE": 365}`。同日可比较完整值：GHCN 0 天，D222 明确数值日 0 天；若为 0，误差、相关和事件命中率均不可估计。
- `HLA_ISDLITE_2013_STATUS = NOT_ACCEPTED`；`HLA_ISDLITE_RECOVERED_DAYS = 0`；`HLA_REMAINING_UNRESOLVED_DAYS = 111`。
- HLA 2013 有效的 1h 降水字段为 0 条；6h 字段中 211 条数值、29 条 trace、2,674 条缺测。五个重点日期中，7 月 30 日 D222=178.2 mm，ISD-Lite 最大完整窗为 39 mm/6h，GHCN=13.7 mm；8 月 12 日 D222=66.9 mm，但同站同一本地日内存在 79 mm/6h，GHCN=100.8 mm。7 月 21 日和 8 月 9 日子日观测支持降雨事件出现，但无法确认 D222 日量级；8 月 12 日的 6h 值还与 D222 日量级不一致。HLA 2013 climate sanity 仍为 SUSPICIOUS。
- `HLA_RAIN_PROVENANCE = PARTIAL`；`HLA_READY_TO_RESUME_010 = FALSE`。HLA 2004 连续空白 blocker 也仍需解决。

### FQA

- `FQA_EXACT_ISDLITE_STATION_FOUND = FALSE`。2013 实际中国站最近 20 个候选均列在 `fqa_station_candidates.csv`；HLA 与这些候选的所有降水字段 occurrence 都保存在 `precip_occurrences_2013.csv`。
- `FQA_BEST_ISDLITE_CANDIDATE = 570830-99999`，距离 84.66102468351102 km；最近站高程差 82.9 m。该站 8,744 条记录中 1h 降水字段全部缺测，只有 107 个 1h/6h数值降水字段，无法完整覆盖任一本地日。站点空间、海拔、完整日数与 overlap QA 见 `fqa_candidate_spatial_qa.csv` 和 `fqa_candidate_overlap_qa.csv`。
- 最近站的完整日总量与 D222 有效 overlap 若不足，湿/干命中、MAE、RMSE、相关和累计差均保留为空值，不把缺测日当干日。D222 空白也未置 0。
- `FQA_ISDLITE_2013_STATUS = REFERENCE_ONLY`；`FQA_2013_RECOVERED_DAYS = 0`；`FQA_2013_RAIN_SANITY = SUSPICIOUS`；`FQA_OVERALL_RAIN_PROVENANCE = FAIL`；`FQA_WEATHER_READY_FOR_WGEN = FALSE`。
- CERN D222 与旧 `HLLCYCFQ降雨数据.xls` 的值链在 009 中有闭合证据，但不视为独立观测。ISD-Lite 与 GHCN 同为 NOAA 产品，可能共享国家/GTS 输入链。最近站也有约 84.7 km 距离，因此只可提供 2013 区域辅助信息。D222 最大事件 2013-05-26 的 204.0 mm 与远站、NASA、GHCN 的同日量级差异明显，说明事件发生不等于 magnitude 获得验证。它不能独立解除 FQA 2011/2012 的总量和 2012-09-02 240 mm 极端量级 blocker。

## 降水累计窗方法

ISD-Lite 时间戳按 UTC 解析并转换到 Asia/Shanghai。1h/6h值作为观测时刻前的累计窗口；先生成完整不重叠覆盖路径，再按唯一数值日总量决定 `VALID_COMPLETE_DAY` 或 `VALID_DRY_DAY`。同一时段上的 1h、6h值从不直接相加。trace、缺测、跨午夜、路径冲突和不完整覆盖分别保留 unresolved 状态。具体字段及窗口假设见 [格式说明](hl_fq_isdlite_data_format_021.md)。

## 最终 gate 与执行范围

`final_gate.json` 已记录 HLA、FQA 状态。没有运行 fitting weather、CLI、WGEN、DSSAT 或 PPO；没有修改 D222、原始压缩包、原始天气文件或现有实验结果。

机器可读汇总：

| 状态 | HLA | FQA |
|---|---|---|
| ISD-Lite 2013 | NOT_ACCEPTED | REFERENCE_ONLY |
| 身份 / exact 站 | PASS | FALSE |
| 缺口起点 / 恢复 / 剩余 | 111 / 0 / 111 | 157 / 0 / 157 |
| Rain provenance | PARTIAL | FAIL |
| 可进入 WGEN 前序工作 | FALSE | FALSE |

## 结果文件

- `results/hl_fq_isdlite_021/input_provenance.json`、`dataset_identity.json`、`final_gate.json`
- HLA identity、逐日数据、GHCN/D222 overlap、111 天恢复和重点事件表
- FQA 最近 20 站、逐站逐日数据、空间/overlap QA、157 天恢复、2013 来源覆盖汇总和极端事件上下文表
- `results/hl_fq_isdlite_021/findings.md`
