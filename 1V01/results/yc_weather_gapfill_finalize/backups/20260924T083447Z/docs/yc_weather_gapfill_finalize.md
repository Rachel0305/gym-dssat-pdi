# YC 变量级天气缺口补齐审计（003_05_01）

**最终状态：** `PASS_YC_WEATHER_CANDIDATE`  
**范围：** YC/YCA，训练天气 2004–2013；本轮未生成 `.CLI`，未运行 WeatherMan、WGEN、DSSAT 或 PPO。

## 1. 变量级 Gate

上一轮的 `accepted_for_gap_fill` 是全变量联合结果；RAIN 未通过导致所有变量一起被拒绝。本轮保持原阈值和上一轮校正参数，只对各变量自己的验证项独立判定，不修改生产数据。

| 变量 | Gate | overlap n | 校正后 MAE | RMSE | Pearson r | NASA 接受填补天数 |
|---|---:|---:|---:|---:|---:|---:|
| TMAX | PASS | 3180 | 1.721 | 2.316 | 0.9808 | 107 |
| TMIN | PASS | 3180 | 1.630 | 2.120 | 0.9814 | 107 |
| SRAD | PASS | 3136 | 1.562 | 2.482 | 0.9363 | 151 |

RAIN overlap 为 3287 天，事件一致 2366 天（71.98%），固定门槛 80.00%；因此 NASA RAIN 继续拒绝。TMAX/TMIN/SRAD 沿用 2005–2013 官方值与 NASA 原值计算的加性偏差校正，不覆盖已有有效官方值。

## 2. 本地日降雨源检查

YCA ChinaFLUX 2004 日尺度产品共 366 日，其中 361 日有有效降雨值；目标五天原值均为缺失码：

| 日期 | ChinaFLUX 日产品原值 | 可用 |
|---|---:|---:|
| 2004-10-16 | -99999 | 否 |
| 2004-10-17 | -99999 | 否 |
| 2004-10-18 | -99999 | 否 |
| 2004-10-19 | -99999 | 否 |
| 2004-10-20 | -99999 | 否 |

项目内自动降水 XLS 与人工气象记录为月统计，不足以还原逐日降水；月/年值不拆分、不反推到五天。

## 3. GHCN-Daily 附近站验证

从 NOAA GHCN-Daily 站点目录按 YC 坐标 36.830°N, 116.570°E 搜索 300 km 内的 12 个站点。PRCP 原始单位为 0.1 mm，按 NOAA 格式除以 10；`-9999` 作为缺测码，非空质量标志不计入有效配对，原始 M/Q/S 标志保留。最低 paired days 采用上一轮 RAIN coverage gate 的 3000 天；事件门槛仍为 80%。Pearson 仅作参考，月/年总量比用于排查处理问题和解释差异。

| Station | 名称 | 距离 km | 配对日 | 事件一致率 | 2004 五天有效数 | Gate |
|---|---|---:|---:|---:|---:|---:|
| CHM00054823 | JINAN | 49.8 | 3039 | 85.32% | 5/5 | PASS |
| CHM00054725 | HUIMIN | 113.3 | 3038 | 86.34% | 5/5 | PASS |
| CHM00054618 | POTOU | 139.3 | 3024 | 85.35% | 5/5 | PASS |
| CHM00054916 | YANZHOU | 142.7 | 3027 | 84.67% | 5/5 | PASS |
| CHM00054616 | CANGZHOU | 168.7 | 0 | NA | 0/5 | FAIL |
| CHM00054906 | HEZE/CAOZHOU | 203.3 | 0 | NA | 0/5 | FAIL |
| CHM00053898 | ANYANG | 212.6 | 3042 | 82.45% | 5/5 | PASS |
| CHM00054909 | DINGTAO | 216.1 | 3019 | 82.05% | 5/5 | PASS |
| CHM00053698 | SHIJIAZHUANG | 232.4 | 3029 | 79.30% | 5/5 | FAIL |
| CHM00054843 | WEIFANG | 232.8 | 3053 | 81.46% | 5/5 | PASS |
| CHM00054527 | TIANJIN | 257.8 | 3029 | 81.68% | 5/5 | PASS |
| CHM00058027 | XUZHOU | 288.0 | 3063 | 74.40% | 5/5 | FAIL |

### 五天决议

| 日期 | 最终降雨 | 站点 | QC |
|---|---:|---|---|
| 2004-10-16 | 0.0 mm | CHM00054823 | ACCEPTED_GHCN_STATION_GATE_PASS |
| 2004-10-17 | 0.0 mm | CHM00054823 | ACCEPTED_GHCN_STATION_GATE_PASS |
| 2004-10-18 | 0.0 mm | CHM00054823 | ACCEPTED_GHCN_STATION_GATE_PASS |
| 2004-10-19 | 0.0 mm | CHM00054823 | ACCEPTED_GHCN_STATION_GATE_PASS |
| 2004-10-20 | 0.0 mm | CHM00054823 | ACCEPTED_GHCN_STATION_GATE_PASS |

`rain_station_target_values.csv` 保存全部候选站对五天的原始值与 flags；NASA POWER 降雨仅列作辅助，不参与选择。NASA POWER 为网格/模式驱动产品，不称为站点实测。[NASA POWER Daily API 文档](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)

## 4. Candidate、气候统计与 QC

- NASA 接受填补：TMAX 107 天，TMIN 107 天，SRAD 151 天；RAIN 0 天。
- 2004 candidate 年降雨：846.2 mm。ChinaFLUX 年尺度产品参照为 846.2 mm；仅作独立年值参照，不用于分配目标五天。
- Candidate 与 ChinaFLUX 年/月产品比较：`已完成，详见 chinaflux_2004_candidate_comparison.csv`。
- Candidate QC：`passed=True`，`not_run=False`；原因：无。
- `yc_wgen_fitting_weather_2004_2013.csv`：未生成；因此 3653 天完整性、全序列物理 QC 和 candidate 年/月统计均未运行。汇总 CSV 明确标为 `NOT_RUN_BLOCKED_FIVE_DAY_RAIN`。
- 三变量已接受缺口合计 365 个 variable-days，占原 370 个待处理 variable-days 的 98.65%；这不代表天气 candidate 完整。

## 5. 泄漏与复现

官方源表只解析到 2013；NASA 偏差校正/验证年份为 2005–2013。GHCN `.dly` 只解析 2004-10-16 至 20 及 2005–2013 的 PRCP；2014+ 行仅参与文件 SHA256，未解析天气值。泄漏审计：`True`，最大用于值/校准年份 2013。

原始来源 URL、检索时间、HTTP 元数据和文件哈希见 `run_manifest.json`；仅把分析期间的逐日观测子集保存于 `rain_station_source_values.csv`，没有缓存/提交全站历史数据。

## 6. 下一步

当前状态 `PASS_YC_WEATHER_CANDIDATE`。最小下一步是为 2004-10-16 至 20 找到可信且满足同一 80% overlap gate 的逐日地面站来源；不得调低门槛或使用月/年总量拆分。只有五天来源通过后，才生成 candidate 并执行完整 QC；本轮不进入 `.CLI`、WGEN、DSSAT 或 PPO。
