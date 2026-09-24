# YC 多源天气重建实验记录（003_04）

## 任务边界

- 仅 YC/YCA，训练天气年限 2004–2013；禁止用 2014+ 天气值。
- 原始源文件只读；派生审计结果只写入 `results/yc_multisource_weather_reconstruction/`。
- 不修改生产 `.WTH`、`.CLI`、FileX、SOL、CUL；不运行 WGEN、DSSAT、PPO。
- 本轮没有生成 `yc_wgen_fitting_weather_2004_2013.csv` 或 candidate `.WTH`。

## 开始状态

- Branch：`codex/sya-forecast-freeze-2026-08-16`
- HEAD：`6677f77a5768d13c77af08c5c378107ceeeba5b6`
- 开始时已存在的 tracked 修改：`configs/068_effective_hla_seed1_smoke.yaml`、`configs/068_effective_yca_seed1_smoke.yaml`、`proposal_materials/东雨淋-博士开题答辩讲稿-20分钟.md`、`src/054_hla_lowIC_site_transfer/run_054_02_hla_lowIC_five_scenario_figures.py`、`src/mask_aware_dqn_029.py`。
- 开始时仓库还有大量其他未跟踪结果/脚本/临时构建目录；均未清理、未改动、未纳入本任务提交。

## 操作过程

1. 阅读 `prompt_02/003_04_yc_multisource_weather_reconstruction.md` 和计划文件，核对 003_03 的 unresolved gap 清单与上一轮温度映射结论。
2. 盘点 `data/external/yc_chinaflux/raw/`。ChinaFLUX 的 30 min、daily、monthly、yearly ZIP/PDF 均存在；还存在 1998–2006 产品 ZIP/PDF、2005–2022 气象与辐射 XLSX 及说明 DOCX。
3. 解析产品说明与归档目录。1998–2006 产品 PDF 确认其时间分辨率为月，ZIP 含 10 个 `.xls`；因此拒绝把其月总量拆成逐日值。
4. 解析 2005–2022 DOCX 和两个 XLSX 的表头。只迭代到 2013 年；登记温度、雨量、辐射字段和缺测样式。说明文档称数据经 EcoFlow 基本 QC、缺测插补和日汇总，但工作簿没有逐日实测/插补标志。
5. 读取 ChinaFLUX 2004–2010 半小时工作簿。只有日内 48 个半小时记录完整时才生成该日降水或总辐射聚合值。温度极值保留为近地面/冠层上方两个并行候选，不做高度映射。
6. 在 2005–2006 重叠期做日值对照，输出相关、偏差、MAE、RMSE、年/月汇总和降雨事件一致性。
7. 对 35 个未解决降水日逐日查 2005–2022 官方日表。30 天有数值（全为 0），但仅保留为暂定替代证据；5 天位于 2004 年，产品无日值覆盖。
8. 对 2011–2013 输出 1,096 个日历日的原始日表候选值和 QC 状态，不填补、不截负值。生成 2004–2013 共 3,653 行变量级 provenance。
9. 第一次检查发现 provenance 组装遗漏 003_03 中 8 个可追溯旧原始降水值，且没有接入 2011–2013 新日表降水列；修正来源选择顺序后重跑。覆盖前将上一版本任务文件备份至 `results/yc_multisource_weather_reconstruction/backups/20260924_085411/`；当前目录顶层文件为修正版，备份快照不提交。
10. 将 003_03 unresolved gap 清单、legacy provenance 清单、旧源派生观测表及清单记录的 3 个旧 Excel SHA256 纳入本轮来源 inventory；原始 Excel 未在本轮重新读取。
11. 代码复核发现月统计曾把 2005/2006 同月份合并，且变长字段导致温度月均值列未写出。修正为逐年逐月 `YYYY-MM` 共 24 期，并使用同时包含累计量和均值的固定字段表头后重跑。

## 错误、纠正与影响

- 一次早期临时统计误把 XLSX Python 索引 20（风向）当成日降水。产生的异常年总量约 60,000，立即因量级不合理停止使用；根据表头“日降水量”确认 Python 索引 22（Excel 第 23 列），重算全部雨量统计。误算未进入最终报告或当前顶层结果。
- `.xls` 读取依赖在运行时不可用（无 `xlrd`、`python-calamine`、`olefile` 等）；本轮未安装软件，也未修改源 ZIP。该产品只能从 PDF/ZIP 结构确认覆盖期、月分辨率及变量，XLS 内部单位行未读，故不用于数值决定。
- 2005–2022 日表实际出现全角破折号 `－`。按缺测处理，不转换为 0；说明文件未明确此编码，已记为元数据缺口。
- 上述月统计修正前的结果文件已分别备份在 `backups/20260924_090626/`、`backups/20260924_091020/`、`backups/20260924_091303/`；当前顶层 CSV/JSON 是最后重跑版本。PPTX 版式修订前的本轮生成文件备份在项目 `backups/yc_multisource_weather_reconstruction_20260924_090657.pptx`。

## 关键证据与决定

- Gate A：2005–2006 有 705 个完整温度配对日。近地面候选对 TMAX/TMIN 的 MAE 分别为 0.8676/0.6128°C；冠层上方分别为 1.0191/0.7710°C。虽然近地面数值误差较小，但日表传感器高度不明，ChinaFLUX 1.6 m/2.9 m 未与导出字段配对；不按标签或数值误差猜高度，状态 `BLOCKED_TEMPERATURE_MAPPING`。
- Gate B：35 个原 unresolved 降水日中，30 个在新官方日表找到数值 0，但官方处理说明包含缺测插补且无逐日状态。2004-10-16 至 20 五日无日尺度源；月值不可拆分。重叠期 2005 年日总量为 ChinaFLUX 627.2 mm、新日表 678.4 mm；2006 年分别为 380.2 mm、403.6 mm。降水日配对 bias +0.1022 mm/d、MAE 0.5597、RMSE 2.5000、r=0.93665，事件一致率 674/730（92.33%）。来源差异原因未被现有元数据解释，30 个候选未准入最终数据，状态 `BLOCKED_PRECIPITATION_GAPS`。
- Gate C：2011 年缺 TMAX/TMIN 38 日、缺 SRAD 36 日，SRAD 负值 2 日；2012 年三者在 06-03 缺 1 日；2013 年 SRAD 在 10-06 缺 1 日，另有 01-02 负值。RAIN 三年均有逐日数值。总辐射字段定义与单位清楚，但不得把缺值当零或裁剪负值，状态 `BLOCKED_2011_2013_WEATHER`。
- 2014+ 泄漏：读取新产品时按日序遇到 2014 即停止，代码含 `assert year <= 2013`；ChinaFLUX 处理仅限 2004–2010。没有 2014+ 天气值进入计算、拟合、插补或参数选择。

## 明确拒绝的处理

- 不把 1998–2006 月降水、月温度或月辐射拆分为逐日值。
- 不把空白、缺测编码或全角破折号自动转成 0；30 个来源表中已有的数值 0 也不视作已证明的实测零值。
- 不对降雨线性插值、不按邻日前后日猜值。
- 不把日平均温度替代 TMAX/TMIN；不把净辐射、PAR 或长波辐射当 SRAD。
- 不按字段中文名、温度曲线相关性或更小 MAE 猜传感器高度。
- 不使用 2014–2023 validation 数据，不覆盖生产天气文件，不运行 WGEN/DSSAT/PPO。

## 当前产物与复跑

机器可读审计、来源清单、差异统计、温度/降水/2011–2013 QC、日级 provenance 均位于本目录。由于门槛未通过，`final_candidate_qc.csv` 与月/年汇总记录 `NOT_RUN_CANDIDATE_NOT_BUILT`，不表示 candidate 已经通过 QC。

默认审计命令：`python scripts/reconstruct_yc_multisource_weather.py --audit-only`。脚本需要 `openpyxl`、`python-docx` 和 `pypdf`；重跑时如需覆盖既有本任务输出，使用 `--overwrite`，脚本会先复制顶层结果到时间戳备份目录。
