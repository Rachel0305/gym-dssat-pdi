# YC ChinaFLUX 气象数据对齐与 WGEN 拟合候选审计

- **任务：** 003_03，站点 YC/YCA，数据期 2003–2010
- **审计结论：** `BLOCKED_TEMPERATURE_SENSOR_MAPPING`
- **次级阻塞：** `BLOCKED_PRECIPITATION_GAPS`

## 结论摘要

四个 ChinaFLUX 压缩包均已只读解析，每个压缩包包含 2003–2010 共 8 个年度 Excel，合计 32 个工作簿。30 分钟数据的时间网格完整连续，没有重复、乱序或缺失时间戳。近地面温度、冠层上方温度和太阳辐射记录无空值或 `-99999`；缺测集中在降水。

当前不能选定唯一的 DSSAT `TMAX/TMIN` 来源：产品字典说明两个温度字段分别位于冠层下方和冠层上方，并列出 1.6 m、2.9 m 两个高度，但没有把导出字段与高度逐一对应。两条日温度候选支路都保留在 CSV，规范 `TMAX/TMIN` 列留空。

ChinaFLUX 30 分钟降水共有 1,193 个缺测半小时记录，分布在 43 天。核对旧原始降雨表后，只有 8 天存在可追溯的数值观测，可作为明确标记的旧源回填；另 35 天仍未解决。2007 年日值降雨年和与其它尺度差 524.54 mm，不能用该日值产品补齐缺测日。

因此 Option A（2004–2010）虽有 7 年，仍有温度映射和降水完整性阻塞；Option B（2004–2013）另含 2011–2013 的异源数据和 125 个 SRAD/温度预填补值。两者均未获准用于 `.CLI` 拟合。本轮未生成正式 `.WTH` 或 `.CLI`，未运行 DSSAT smoke 或 PPO。

## 数据来源与范围

输入压缩包位于 `data/external/yc_chinaflux/raw/`：

| 压缩包 | 内部年度工作簿 | 年份 |
|---|---:|---|
| `YCA_M_30min.zip` | 8 | 2003–2010 |
| `YCA_M_daily.zip` | 8 | 2003–2010 |
| `YCA_M_monthly.zip` | 8 | 2003–2010 |
| `YCA_M_yearly.zip` | 8 | 2003–2010 |

文件字节数、SHA256、内部 Excel 文件名及解析状态记录于 [`source_inventory.json`](../results/yc_chinaflux_weather_reconciliation/source_inventory.json)。原始 ZIP 未解压、改写或移动。旧项目源表 `T2.xls`、`D32.xls` 和 YC 降雨工作簿通过 Excel COM 只读打开，保留原始空值状态；字段映射和文件哈希记录在 [`legacy_source_inventory.json`](../results/yc_chinaflux_weather_reconciliation/legacy_source_inventory.json)。

产品字段、单位及传感器说明见 [30 分钟产品说明](../data/external/yc_chinaflux/raw/YCA_M_30min.pdf)、[日尺度产品说明](../data/external/yc_chinaflux/raw/YCA_M_DAILY.pdf)、[月尺度产品说明](../data/external/yc_chinaflux/raw/YCA_M_MONTHLY.pdf) 和 [年尺度产品说明](../data/external/yc_chinaflux/raw/YCA_M_YEARLY.pdf)。ChinaFLUX 禹城站资源页：[国家生态科学数据中心](https://www.nesdc.org.cn/otherProject/index?menuId=station&projectId=1047)。产品说明标注已进行质量控制/插补，因此“非缺测”不等同于未经处理的现场原始观测。

## 降水跨尺度核验

30 分钟列的年和包含不完整日期上的有效半小时值，只能作为下界；日、月、年产品有效值按各自时间尺度直接求和。所有值单位为 mm。

| 年份 | 30 分钟有效值和（下界） | 日值有效和 | 月值和 | 年值 | 旧 WTH 年和 |
|---:|---:|---:|---:|---:|---:|
| 2003 | 697.1 | 697.1 | 697.1 | 697.1 | 不在本轮 WGEN 比较窗 |
| 2004 | 846.3 | 846.2 | 846.2 | 846.2 | 67.4 |
| 2005 | 627.2 | 627.2 | 627.2 | 627.2 | 678.4 |
| 2006 | 380.2 | 380.2 | 380.2 | 380.2 | 381.8 |
| 2007 | 535.7 | **11.16**（364 个有效日） | 535.7 | 535.7 | 571.7 |
| 2008 | 477.9 | 477.9 | 477.9 | 477.9 | 527.1 |
| 2009 | 733.7 | 733.7 | 733.7 | 733.7 | 802.7 |
| 2010 | 186.7（下界） | 186.5（349 个有效日） | 186.5 | 186.5 | 739.9 |

2004 年旧 `CNYC0401.WTH` 年降水为 67.4 mm。ChinaFLUX 日、月、年产品均为 846.2 mm；30 分钟有效记录和为 846.3 mm，但当年有 186 个缺测半小时记录分布于 5 天，不能把 846.3 表述为完整精确总量。新旧差异与旧降雨空值被预处理转为 0 的记录证据一致。

2007 年日值产品年和仅 11.16 mm，而 30 分钟有效值和、月值和及年值均约 535.7 mm，日值相差 −524.54 mm。按任务规则，日值降水只保留作参考，不能作为统一主源或用来回填日缺测；月、年总量也不向日尺度分配。逐年结果见 [`cross_scale_precipitation_check.csv`](../results/yc_chinaflux_weather_reconciliation/cross_scale_precipitation_check.csv)。

## 30 分钟完整性与日值转换

每年预期记录数为平年 17,520、闰年 17,568。八年均达到预期数，半小时网格连续、唯一且按时间递增。30 分钟字段按 `近地面空气温度`、`冠层上方空气温度`、`太阳辐射`、`降水量` 分别审计；没有把 `-99999` 视为 0。

| 变量 | 审计结果 |
|---|---|
| 两个温度字段 | 140,256 条记录均有效，无空值、无 `-99999` |
| 太阳辐射 | 140,256 条记录均有效，无空值、无 `-99999` |
| 降水 | 1,193/140,256 个半小时值为 `-99999`；43 个日期不完整，缺测集中于 2004、2007–2010 |
| 时间戳 | 0 重复、0 缺失网格位置、0 乱序步长 |

从 30 分钟平均辐射通量计算 DSSAT 日辐射：`SRAD = sum(W m-2) × 0.0018`。以日尺度产品的日平均辐射计算独立核验：`SRAD_check = daily_mean(W m-2) × 0.0864`。2003–2010 共 2,922 天，两种换算的 MAE 和最大绝对差均为 0 MJ m⁻² d⁻¹。逐日记录在 [`srad_daily_crosscheck.csv`](../results/yc_chinaflux_weather_reconciliation/srad_daily_crosscheck.csv)。

每天温度有效时段均为 48 条。两支路各自以当日最大/最小值构造 `TMAX/TMIN` 候选；在映射确认前，候选文件的规范 `TMAX/TMIN` 列为空，支路列另行保留。

## 温度传感器映射

| 导出字段 | 产品说明的物理位置 | 可用高度证据 | 当前决定 |
|---|---|---|---|
| `近地面空气温度` | 冠层下方 | 产品同时列出 1.6 m、2.9 m，未与本字段配对 | 保留候选，不选为规范温度 |
| `冠层上方空气温度` | 冠层上方 | 产品同时列出 1.6 m、2.9 m，未与本字段配对 | 保留候选，不选为规范温度 |

公开资料不能补足字段映射：[AsiaFlux 禹城站页面](https://asiaflux.nies.go.jp/site-info/YCS.html)列出常规气象气温 1 m、2 m（HMP45C），而 2003–2005 网络论文也描述 1 m、2 m 观测；这些资料没有说明当前导出字段分别对应何传感器或高度。字段名、导出顺序和高相关性都不足以确定 DSSAT 应采用哪条支路。需由数据提供方确认字段到高度的配对及其统计口径。

## 新旧天气逐日比较

新旧对比仅用于差异审计，不把旧 WTH 视为真值。温度和 SRAD 各有 2,557 个成对日值。旧源若标为月均值填补、降雨空白转零或无源行，仍参与描述性对比，但不能据此证明测量一致。

| ChinaFLUX 温度支路 | 指标 | TMAX | TMIN |
|---|---|---:|---:|
| 冠层下近地温度 | Bias / MAE / RMSE（°C） | −0.692 / 1.017 / 1.481 | −0.234 / 0.710 / 1.287 |
| 冠层下近地温度 | Pearson r | 0.9927 | 0.9927 |
| 冠层上方温度 | Bias / MAE / RMSE（°C） | −0.791 / 1.080 / 1.514 | +0.167 / 0.848 / 1.298 |
| 冠层上方温度 | Pearson r | 0.9928 | 0.9924 |

SRAD（2,557 天）：Bias −0.484、MAE 0.949、RMSE 1.749 MJ m⁻² d⁻¹，Pearson r=0.9687。降雨只在 ChinaFLUX 30 分钟全天完整且未使用旧源回填的日期比较，共 2,514 天；逐日、逐月及雨日统计见 [`old_vs_chinaflux_summary.csv`](../results/yc_chinaflux_weather_reconciliation/old_vs_chinaflux_summary.csv) 和 [`old_vs_chinaflux_monthly_rain.csv`](../results/yc_chinaflux_weather_reconciliation/old_vs_chinaflux_monthly_rain.csv)。所有逐日配对值及旧源出处标记见 [`old_vs_chinaflux_daily_comparison.csv`](../results/yc_chinaflux_weather_reconciliation/old_vs_chinaflux_daily_comparison.csv)。

旧源分类基于只读原始表、旧清洗代码和现有 WTH 的数值对照，分类为原始数值与 WTH 一致、旧流程月均值填补、降雨空白/无效值转 0、旧原始数值与 WTH 不一致，以及无法判断。2009 年 12 月 1–31 日三个旧气象源工作簿均没有 YCA 行，单独标记为 `no_source_row`，没有伪装为空白观测或零值。

## 未解决缺口与拟合窗口

| 年份 | ChinaFLUX 降雨不完整日 | 可追溯旧日值回填 | 仍未解决 |
|---:|---:|---:|---:|
| 2004 | 5 | 0 | 5 |
| 2007 | 3 | 0 | 3 |
| 2008 | 3 | 0 | 3 |
| 2009 | 6 | 1 | 5 |
| 2010 | 26 | 7 | 19 |
| 合计 | 43 | 8 | 35 |

2007 日值降雨年总量本身存在严重跨尺度异常，所以即使个别缺测日期有日值数字，也没有将其认作可信回填。2009 年 12 月旧源整月无行，且 ChinaFLUX 日降雨也缺测的日期仍保持缺失。每个日期及 30 分钟缺测数、日/月/年参考值、旧原始状态和处理决定均列于 [`unresolved_weather_gaps.csv`](../results/yc_chinaflux_weather_reconciliation/unresolved_weather_gaps.csv)；半小时连续缺测明细见 [`missing_daily_intervals.csv`](../results/yc_chinaflux_weather_reconciliation/missing_daily_intervals.csv)。

| 拟合窗口 | 年数 | 证据与判断 | 当前资格 |
|---|---:|---|---|
| Option A：2004–2010 | 7 | 都在原训练年份；35 天降雨仍缺，8 天需明确标注旧源回填；温度高度映射未决 | 不可生成 CLI |
| Option B：2004–2013 | 10 | 2011–2013 只靠旧数据，存在 125 个 SRAD/温度源缺口值并曾按月均值预填；来源不一致 | 不可生成 CLI |

DSSAT WeatherMan FAQ 提到导入至少 5–10 年日天气作为气候资料输入的常规指导，但这只是记录年限参考，不豁免变量完整性、来源一致性或温度映射门槛。[DSSAT WeatherMan FAQ](https://dssat.net/5165/)。具体判断见 [`wgen_fit_window_options.json`](../results/yc_chinaflux_weather_reconciliation/wgen_fit_window_options.json)。本轮未使用 validation/test 年份。

## Gate 与下一步

最终状态为 `BLOCKED_TEMPERATURE_SENSOR_MAPPING`，并记录次级状态 `BLOCKED_PRECIPITATION_GAPS`。两套温度支路、候选日数据和全部审计表都保留，未静默补齐缺测，未选择 WGEN 拟合窗口。恢复流程前至少需要：

1. 请 ChinaFLUX/数据提供方明确当前两个温度导出字段分别对应的传感器高度及统计定义。
2. 获取 35 个未解决降水日期的独立可追溯日观测，并核实 2010 ChinaFLUX 与旧站点年总量差异；不得按月/年总量分摊。
3. 在用户确认温度字段和拟合窗口后，重跑完整 QC Gate，再讨论 `.CLI` 生成；本报告不授权进入后续模拟或 PPO。

## 可复现脚本与关键结果

- [`reconcile_yc_chinaflux_legacy_excel.ps1`](../scripts/reconcile_yc_chinaflux_legacy_excel.ps1)：只读提取原始旧源；按表头定位 `20-20合计` 降雨列。
- [`reconcile_yc_chinaflux_weather.py`](../scripts/reconcile_yc_chinaflux_weather.py)：直接读取 ZIP 内 Excel，完成多尺度重建、质量核验、候选和比较输出。
- [`audit_summary.json`](../results/yc_chinaflux_weather_reconciliation/audit_summary.json)：Gate、数据范围和阻塞摘要。
- [`yc_chinaflux_daily_candidate_2003_2010.csv`](../results/yc_chinaflux_weather_reconciliation/yc_chinaflux_daily_candidate_2003_2010.csv)：2003–2010 双温度支路日候选。
- [`yc_wgen_candidate_2004_2010.csv`](../results/yc_chinaflux_weather_reconciliation/yc_wgen_candidate_2004_2010.csv)：2004–2010 拟合候选，不是正式 DSSAT 天气文件。

复现顺序：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reconcile_yc_chinaflux_legacy_excel.ps1
& 'C:\Users\DELL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/reconcile_yc_chinaflux_weather.py
```

所有分析输出位于独立结果目录，没有覆盖正式 `.WTH`、`.CLI`、FileX 或历史实验目录。
