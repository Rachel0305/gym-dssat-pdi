# YC 多源天气重建审计（003_04）

**审计范围：** YC/YCA；拟用天气期 2004–2013。

**最终状态：** `BLOCKED_TEMPERATURE_MAPPING`。降水缺口、2011–2013 完整性和跨产品一致性同时未通过。

**候选数据：** 未生成 `yc_wgen_fitting_weather_2004_2013.csv`；未生成生产 `.WTH`/`.CLI`，未运行 WGEN、DSSAT 或 PPO。

## 1. 结论摘要

三类数据产品均已盘点，但可支持的分辨率和 QC 状态不同。1998–2006 禹城站监测产品是月值统计，不能作为日降水缺口的日值来源；2005–2022 产品提供日尺度气象和辐射，但其说明记录了产品级缺测插补，工作簿没有逐日“实测/插补”标志。ChinaFLUX 30 分钟产品温度字段写明冠层下/冠层上，传感器安装高度只并列给出 1.6 m 和 2.9 m，未将高度配到导出字段。

| 门槛 | 状态 | 关键依据 |
|---|---|---|
| Gate A 温度字段/高度映射 | `BLOCKED_TEMPERATURE_MAPPING` | 重叠数值略偏向“近地面”字段，但比较用日表没有传感器高度，不能确认 ChinaFLUX 两列与 1.6/2.9 m 的对应关系。 |
| Gate B 35 个降水缺口 | `BLOCKED_PRECIPITATION_GAPS` | 新日表为 30 天提供数值候选（均为 0），但无逐日实测/插补标志；5 个 2004 日期没有可用日尺度来源。 |
| Gate C 2011–2013 | `BLOCKED_2011_2013_WEATHER` | 降水日值完整；TMAX/TMIN、SRAD 存在缺测，SRAD 另有负值。 |
| Source priority | `BLOCKED_SOURCE_INCONSISTENCY` | 2005、2006 年降水年总量差异分别为 51.2 mm、23.4 mm，现有元数据无法判定测量/处理口径差异。 |

未以“站点相同”推导“观测独立”。报告中的重叠比较均称为同站点多产品一致性检查，不视为独立验证。

## 2. 数据产品与来源

来源文件及 SHA256、内部归档成员、变量、单位、缺测标记、分辨率和观测状态见 [`source_inventory.csv`](../results/yc_multisource_weather_reconstruction/source_inventory.csv) 与 [`source_metadata.json`](../results/yc_multisource_weather_reconstruction/source_metadata.json)。其中也登记了 003_03 缺口清单、旧源 provenance 派生表和旧源清单哈希；3 个旧 `.xls` 原始文件哈希从上一轮清单继承，本轮未重新读取这些来源文件。输入原件保留在 `data/external/yc_chinaflux/raw/`，未解压覆盖或编辑。

| 产品 | 实际覆盖/分辨率 | 变量与处理状态 | 本轮可用性 |
|---|---|---|---|
| ChinaFLUX YCA 2003–2010 | 30 min、日、月、年四种产品 | 温湿、降水、辐射等；产品说明为质控/插补后数据；30 min 缺测编码 `-99999`。 | 30 min 可在完整 48 个半小时记录时聚合日雨量、极值候选和总辐射；日/月/年产品仅作核对。2007 日值降水异常，不作主补值。 |
| 1998–2006 禹城站气象和太阳辐射监测数据 | 1998-01 至 2006-12，月值；ZIP 含 10 个 `.xls` 与关联 PDF。 | 自动/人工气象及辐射月统计。产品说明确认月分辨率；不是逐日日值。 | 可作月尺度背景，不能拆分为日降水、TMAX/TMIN 或 SRAD。嵌入 `.xls` 未由当前运行时的 XLS 解析器读取，单位/列标题按产品 PDF 未能完整确认，故没有用其数值。 |
| 禹城站 2005–2022 大气环境要素数据集 | 2005–2022，日值；气象与辐射各一份 XLSX。 | 气象列含日平均/最高/最低温度、降水；辐射列含总辐射、净辐射、PAR。说明称逐时观测经 EcoFlow 基础 QC、缺测插补及日汇总；未提供逐单元格的实测/插补标志。 | 2011–2013 用于日尺度候选审计；值读取严格截止 2013。`－`（U+FF0D）在工作簿中作为缺测标记出现，说明文档未明确列出该编码。 |

2005–2022 产品说明列出气象站传感器型号及 2014 年 5 月设备更新，但未给空气温度传感器安装高度。1998–2006 数据产品不是可替代的每日序列。因缺少安全可用的 `.xls` 解析依赖，本轮只解析其产品 PDF、压缩包成员结构和时间分辨率；未安装软件，也未将源文件解压到其他目录。

## 3. Gate A：温度字段映射

ChinaFLUX 说明将“近地面空气温度”定义为植被冠层下方空气温度，将“冠层上方空气温度”定义为冠层上方空气温度；设备表对空气温度列出 1.6 m、2.9 m 两个高度，但没有逐字段配对。2005–2022 日表含明确的日最高/最低空气温度，但高度未注明。因而无法把比较序列当作已知 2 m 序列。

2005–2006 完整重叠期数值比较如下。偏差定义为“2005–2022 日表值 − ChinaFLUX 30 min 日极值”。两产品同站点，相关性只说明数值一致程度，不能证明传感器高度或独立性。

| ChinaFLUX 候选字段 | 对比变量 | 配对日数 | Bias (°C) | MAE (°C) | RMSE (°C) | Pearson r |
|---|---:|---:|---:|---:|---:|---:|
| 近地面空气温度 | TMAX | 705 | 0.8066 | 0.8676 | 1.0911 | 0.99778 |
| 近地面空气温度 | TMIN | 705 | 0.1524 | 0.6128 | 1.0510 | 0.99528 |
| 冠层上方空气温度 | TMAX | 705 | 0.9626 | 1.0191 | 1.2022 | 0.99788 |
| 冠层上方空气温度 | TMIN | 705 | -0.2275 | 0.7710 | 1.0968 | 0.99500 |

数值误差在“近地面”列更小，但并不能据此认定它对应 1.6 m，更不能消除对比日表高度未知这一限制。未选择 TMAX/TMIN 主来源，Gate A 阻塞。逐项证据见 [`temperature_mapping_evidence.csv`](../results/yc_multisource_weather_reconstruction/temperature_mapping_evidence.csv) 和 [`temperature_sensor_mapping.json`](../results/yc_multisource_weather_reconstruction/temperature_sensor_mapping.json)。

## 4. Gate B：35 个降水缺口

起点为 003_03 `unresolved_weather_gaps.csv` 中的 35 个 `RAIN` 日期。

| 年份 | 缺口数 | 2005–2022 日表候选 |
|---:|---:|---|
| 2004 | 5 | 无覆盖；`1998–2006` 来源仅月值，禁止拆分。 |
| 2007 | 3 | 3 个数值候选，均为 0 mm。 |
| 2008 | 3 | 3 个数值候选，均为 0 mm。 |
| 2009 | 5 | 5 个数值候选，均为 0 mm。 |
| 2010 | 19 | 19 个数值候选，均为 0 mm。 |

因此，30 天有官方日产品数值可查，5 天（2004-10-16 至 2004-10-20）没有日尺度来源。但不能把这 30 个 0 当作已经认证的实测零值：数据说明确认产品执行过缺测插补，工作簿没有逐日 QC/插补标志；此外，2005–2006 重叠期年降水量与 ChinaFLUX 30 min 聚合不一致。审计表保留这 30 个值作为**暂定替代证据**，`selected_value` 不代表已准入最终 candidate。本轮通过 Gate B 的认证日数为 0/35，未决日数仍按 35 计。

2005–2006 重叠对照：

| 年份 | ChinaFLUX 30 min 完整日聚合 (mm) | 2005–2022 日表 (mm) | 差值 (mm) |
|---:|---:|---:|---:|
| 2005 | 627.2 | 678.4 | +51.2（+8.2%） |
| 2006 | 380.2 | 403.6 | +23.4（+6.2%） |

730 个日配对的偏差（新日表−ChinaFLUX）为 +0.1022 mm/d，MAE 0.5597 mm/d，RMSE 2.5000 mm/d，Pearson r=0.93665；按 `RAIN > 0` 计，事件一致为 674/730 天（92.33%）。月总量和月差异见 [`source_overlap_metrics.csv`](../results/yc_multisource_weather_reconstruction/source_overlap_metrics.csv)。现有材料不能区分雨量计、时间边界、QC 与插补对差异的贡献，因此不指定唯一优先源。

此前 8 个可追溯旧原始降水值仍按 `legacy_raw_observation` 保存于日级 provenance；没有把空白转为 0，也没有用月/年总量拆分、邻日插值或 ChinaFLUX 日值异常产品补造数据。逐日决策见 [`precipitation_gap_resolution.csv`](../results/yc_multisource_weather_reconstruction/precipitation_gap_resolution.csv)。

## 5. Gate C：2011–2013 日天气

2005–2022 日表覆盖三个目标年份的全部日历日期；逐日降水均为数值，但 TMAX/TMIN 与总辐射存在下列问题：

| 年份 | 日数 | RAIN 缺失 | TMAX/TMIN 缺失 | SRAD 缺失 | 负 SRAD 日期 |
|---:|---:|---:|---:|---:|---|
| 2011 | 365 | 0 | 38 | 36 | 2011-03-09、2011-04-20 |
| 2012 | 366 | 0 | 1 | 1 | 无 |
| 2013 | 365 | 0 | 0 | 1 | 2013-01-02 |

温度缺口：2011-03-09、2011-07-27、2011-05-18 至 06-03、2011-08-17 至 09-04；2012-06-03。

辐射缺口：2011-05-18 至 06-03、2011-08-17 至 09-04、2012-06-03、2013-10-06。

辐射数据字典把“总辐射”定义为日总量，单位 MJ/m²，且与净辐射、PAR 分列，仪器表列出 CM11 总辐射表；因此变量定义与 DSSAT 入射短波的语义相符，未发现把净辐射或 PAR 当 SRAD 的情况。ChinaFLUX 半小时“太阳辐射”以 W/m² 给出，本轮只在 48 个时间记录均有效时按 `Σ(W/m² × 1800 s) / 1,000,000` 转为 MJ/m²/d。新日表的负值没有被截为 0，也没有用月值、净辐射或 PAR 替代。缺测和负值使 Gate C 阻塞。逐日记录与 QC 见 [`weather_2011_2013_reconstruction.csv`](../results/yc_multisource_weather_reconstruction/weather_2011_2013_reconstruction.csv) 和 [`weather_2011_2013_qc.json`](../results/yc_multisource_weather_reconstruction/weather_2011_2013_qc.json)。

## 6. 日级 provenance 与泄漏检查

[`yc_weather_provenance_daily_2004_2013.csv`](../results/yc_multisource_weather_reconstruction/yc_weather_provenance_daily_2004_2013.csv) 有 3,653 行，覆盖 2004-01-01 至 2013-12-31，无重复日期。降水来源计数：ChinaFLUX 完整半小时日 2,514 天、2005–2022 官方日表暂定值 1,126 天、可追溯旧原始值 8 天、无值 5 天。2011–2013 的降水源为 2005–2022 日表。2014+ 天气值未用于计算、比较、汇总或插补；脚本读取新 XLSX 时在首个 2014 行停止并断言年份不超过 2013。源文件整体 SHA256 仅用于来源完整性，不读取验证期单元格值。

## 7. Final Gate 与下一步

`final_status = BLOCKED_TEMPERATURE_MAPPING`；次级阻塞为 `BLOCKED_PRECIPITATION_GAPS`、`BLOCKED_2011_2013_WEATHER` 和 `BLOCKED_SOURCE_INCONSISTENCY`。[`final_candidate_qc.csv`](../results/yc_multisource_weather_reconstruction/final_candidate_qc.csv) 及月/年汇总文件明确记录为 `NOT_RUN_CANDIDATE_NOT_BUILT`，不是空白的通过结果。

下一步最小任务：向数据集维护方核实 ChinaFLUX 近地面/冠层上方温度列与 1.6/2.9 m 的配对；索取 2005–2022 日产品逐日实测/插补 QC 标志及降水口径说明；为 2004 五个缺口日寻找有明确日期的日尺度观测；补齐 2011–2013 TMAX/TMIN/SRAD 缺测并核查负辐射值。拿到证据后重新运行本脚本，再由全部 Gate 决定是否进入 003_05。当前不生成 `.CLI`，不运行 WGEN/DSSAT/PPO。

## 8. 可复现产物

- 审计脚本：[`reconstruct_yc_multisource_weather.py`](../scripts/reconstruct_yc_multisource_weather.py)
- 机器可读摘要：[`audit_summary.json`](../results/yc_multisource_weather_reconstruction/audit_summary.json)
- 实验记录：[`experiment_log.md`](../results/yc_multisource_weather_reconstruction/experiment_log.md)
- 跨产品统计：[`source_overlap_metrics.csv`](../results/yc_multisource_weather_reconstruction/source_overlap_metrics.csv)

默认模式为 audit-only：`python scripts/reconstruct_yc_multisource_weather.py --audit-only`。只有所有 Gate 通过，`--build-candidate` 才会写出 fitting CSV；若覆盖既有本任务结果，必须显式加 `--overwrite`，脚本会先备份本任务已有顶层文件。
