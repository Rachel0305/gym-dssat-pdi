# GPCC 与 CPC 产品身份核查（Task 023）

## GPCC Full Data Daily v2022

本次文件头身份为 GPCC Full Data Daily Product Version 2022、1.0°、日尺度、陆地雨量计格点产品。文件说明其根据国家气象/水文部门、区域和全球资料集及 WMO GTS 雨量资料构建：将站点日/月份相对降水异常插值后叠加在 GPCC 月产品的月总量上。它不是海伦或封丘单站逐日观测。变量包含 `precip`（mm/day）、`interpolation_error`（mm/day）和 `numgauge`（gauges per grid cell）；缺测值为 -9999。三个本地年度文件分别覆盖 2011、2012、2013，CDF-1 / gzip；经纬度网格中心、变量属性、时间单位、SHA256 见 `input_provenance.json`。GPCC 官方产品页说明 1°、1982–2020、gzip NetCDF 和数据来源：[GPCC Full Data Daily v2022](https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-daily_v2022_doi_download.html)。

## CPC Global Precipitation

本次文件头身份为 `CPC GLOBAL PRCP V1.0 RT`，`precip(time,lat,lon)` 单位 mm、日总量，0.5°全球陆地区域网格。2011–2013 属 2006 年后的 real-time 产品时段；本地文件的变量/属性、时间轴和压缩块信息见 `input_provenance.json`。该 NetCDF 文件只含降水及坐标/时间变量，没有 gauge-count 或质量辅助变量。CPC README 说明分析基于统一雨量计来源与客观分析，2006 年后为 RT 版本，空间分辨率 0.5°；还明确写明各国日累积结束时刻不同，所附 EOD 信息有限且未经充分核验：[CPC Global Daily README](https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/DOCU/PRCP_CU_GAUGE_V1.0GLB_0.50deg_README.txt)。NOAA PSL 也将该产品列为 0.5° 日降水格点产品：[NOAA PSL gridded precipitation catalog](https://psl.noaa.gov/data/gridded/tables/att.precipitation.html)。

## 来源独立性与日界

GPCC 与 CPC 都是格点产品，不是 HLA/FQA 本站观测，也不能视为完全独立观测。GPCC 明示输入包括国家资料和 GTS；CPC 的全球统一分析也整合多来源雨量站资料并使用 GTS 站报告，因此底层站源可能重叠。两产品相互一致只能说明区域产品之间的一致程度，不能替代单站观测核查。

GPCC 时间坐标采用 `days since YYYY-1-1 00:00:00`、`proleptic_gregorian`；CPC 采用 `hours since 1900-01-01 00:00:00` 且坐标标为 `start`。NetCDF 中的 00:00 是日期标签，不足以证明统一 UTC 观测日界。CPC README 进一步指出日累积结束时刻因国家而异。输出保留原始日期；±1 日仅作为对齐敏感性诊断，未平移产品或 D222。

## 网格提取

每个站点、每个产品均按项目 WTH 经纬度选取球面距离最近的格点作为主分析，并同时报告周边 3×3 的均值、中位数、最小值、最大值。距离和格点中心随逐日产品 CSV 输出。3×3 仅作空间背景，未按与 D222 的相似度挑选格点。

本地文件固定为 v2022 GPCC 与 V1.0 RT CPC。报告期间没有更新、复制、解压落地或改写任何原始文件。

## 本次使用版本和经纬度表示

GPCC 官方产品页面说明 v2022 版本资料期为 1982–2020；该页面目前也列出 v2025。本次遵守本地审计输入冻结要求，使用提供的 v2022 文件，不更新产品版本。CPC Global Daily 资料从 1979 年开始，2006 年后使用 RT 版本；2011–2013 年文件属于该时段。两产品均以规则经纬度网格中心表达格点位置，维度、坐标值、缺测、单位和时间轴均记录在 `input_provenance.json`。GPCC 含插值误差及格点雨量计数辅助变量；本次 CPC NetCDF 未发现 gauge count 或质量控制辅助变量。
