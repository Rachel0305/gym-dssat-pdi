# YC 天气定稿实验记录（003_05）

## 开始状态

- Branch / HEAD 由任务启动记录：`codex/sya-forecast-freeze-2026-08-16` / `01b9a9a0f81be746d3ed84d0e7db4cd8a81b0657`。
- 当时 tracked 用户修改：`configs/068_effective_hla_seed1_smoke.yaml`、`configs/068_effective_yca_seed1_smoke.yaml`、博士开题讲稿、HL 图表脚本、`src/mask_aware_dqn_029.py`；另有大量未跟踪内容。本任务没有编辑或提交它们。
- 003_04 已提交，状态为 `BLOCKED_TEMPERATURE_MAPPING` 等；本轮遵循新授权，接受官方 QC 日产品，并采用一致性选择。

## 方法和运行

- 2004 使用 ChinaFLUX 2004 30 min ZIP，逐日要求 48 条有效半小时记录；TMAX/TMIN 只用“近地面空气温度”并标注 consistency-based selection，SRAD 积分，RAIN 求和。
- 2005–2013 使用官方 QC 日产品；0 mm RAIN 按发布产品值保留。
- gap filling 只对缺测、非法值、TMAX<TMIN、2004 30 min 不完整日尝试；NASA POWER 仅为缺口服务。
- 外部请求状态：`cache_reused`；如果有错误：`无`。
- 外部验证 Gate：`False`；Candidate QC：`False`；Final Gate：`BLOCKED_EXTERNAL_GAPFILL`。
- 本轮没有启动 WGEN、大规模天气随机生成、DSSAT 或 PPO；未修改生产 `.WTH`、`.CLI`、FileX、SOL、CUL。

## 数据源哈希

- ChinaFLUX 2003-2010 30min product: `data\external\yc_chinaflux\raw\YCA_M_30min.zip`, SHA256 `0ef898b6718c63cfa91a52cdb66cf8a0df53c0b245ee2a237e708281b39faee4`, cell-value years `2004-2010; 2005-2010 only for source-total comparison`
- Yucheng 2005-2022 official meteorology daily QC product: `data\external\yc_chinaflux\raw\禹城站2005-2022年大气环境要素观测数据集地面气象观测数据.xlsx`, SHA256 `9ed0e2347ad33b59262774d3cfc84b307444b8e59927ef2060a1281663cc589e`, cell-value years `2005-2013 only`
- Yucheng 2005-2022 official radiation daily QC product: `data\external\yc_chinaflux\raw\禹城站2005-2022年大气环境要素观测数据集辐射观测数据.xlsx`, SHA256 `203b10a343ad8dbc88f52df3ad906a650486b0a9237a3679510a95f2fccbe607`, cell-value years `2005-2013 only`

## 关键产物

- 原始 gap 日期：`gaps_before_external_fill.csv`
- overlap validation：`external_gapfill_validation.json`
- 外部填补值：`external_gapfill_values.csv`
- candidate QC / climate summary：`candidate_qc.json`、`candidate_annual_summary.csv`、`candidate_monthly_summary.csv`
- 2014+ leakage：`leakage_audit.json`
- 最终报告：`docs/yc_weather_finalize_and_cli.md`
- PPT：`docs/yc_weather_finalize_and_cli.pptx`，结构检查结果见 `pptx_validation_summary.json`。

## 复核

- Python 版本：`3.12.3`；openpyxl 版本：`3.1.5`。
- 2014+ 断言：代码和清单均要求 `max(source_weather_years_used_for_values_or_calibration) <= 2013`。
- 结果没有宣称 external gap-fill 为实测值；NASA POWER overlap 是同站点产品比较，指标及所有补值来源均保留。

## PPTX 生成与检查

- 文件：`docs/yc_weather_finalize_and_cli.pptx`，6 页；SHA256 `18293787e9c29c9178232b89d09bd32f5d83f6be7c4b2d56eb78925ca6007db0`。
- 结构检查：ZIP 完整=True；所有形状在画布内=True。
- 未执行图像级渲染；当前只确认结构与边界，视觉渲染状态记为 unavailable。
