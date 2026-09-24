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

从 NOAA GHCN-Daily 站点目录按 YC 坐标 36.830°N, 116.570°E 搜索 300 km 内的 12 个站点。PRCP 原始单位为 0.1 mm，按 NOAA 格式除以 10；`-9999` 作为缺测码，非空质量标志不计入有效配对，原始 M/Q/S 标志保留。最低 paired days 采用上一轮 RAIN coverage gate 的 3000 天；事件门槛仍为 80%。月/年总量比只汇总至少 90% 日覆盖的期间；覆盖不全的期间保留逐期数据但排除在汇总之外，避免把部分月/年误作完整聚合量。

| Station | 名称 | 距离 km | 配对日 | 事件一致率 | Precision | Recall | 日 MAE mm | 湿日 MAE mm | Pearson r | 完整月比中位数 (n) | 完整年比中位数 (n) | 五天 | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CHM00054823 | JINAN | 49.8 | 3039 | 85.32% | 52.91% | 94.98% | 1.693 | 8.252 | 0.651 | 1.219 (n=83) | 1.256 (n=8) | 5/5 | PASS |
| CHM00054725 | HUIMIN | 113.3 | 3038 | 86.34% | 55.27% | 85.69% | 1.873 | 9.486 | 0.496 | 0.925 (n=83) | 0.940 (n=8) | 5/5 | PASS |
| CHM00054618 | POTOU | 139.3 | 3024 | 85.35% | 53.59% | 74.39% | 2.214 | 11.004 | 0.354 | 0.810 (n=83) | 0.927 (n=8) | 5/5 | PASS |
| CHM00054916 | YANZHOU | 142.7 | 3027 | 84.67% | 51.49% | 81.72% | 2.560 | 12.290 | 0.339 | 1.291 (n=83) | 1.404 (n=8) | 5/5 | PASS |
| CHM00054616 | CANGZHOU | 168.7 | 0 | NA | NA | NA | NA | NA | NA | NA (n=0) | NA (n=0) | 0/5 | FAIL |
| CHM00054906 | HEZE/CAOZHOU | 203.3 | 0 | NA | NA | NA | NA | NA | NA | NA (n=0) | NA (n=0) | 0/5 | FAIL |
| CHM00053898 | ANYANG | 212.6 | 3042 | 82.45% | 47.29% | 78.37% | 2.058 | 9.723 | 0.431 | 0.870 (n=83) | 0.863 (n=8) | 5/5 | PASS |
| CHM00054909 | DINGTAO | 216.1 | 3019 | 82.05% | 46.55% | 74.59% | 2.504 | 11.745 | 0.296 | 1.230 (n=83) | 1.231 (n=8) | 5/5 | PASS |
| CHM00053698 | SHIJIAZHUANG | 232.4 | 3029 | 79.30% | 42.11% | 73.17% | 2.445 | 10.342 | 0.235 | 0.864 (n=83) | 0.932 (n=8) | 5/5 | FAIL |
| CHM00054843 | WEIFANG | 232.8 | 3053 | 81.46% | 46.06% | 79.92% | 2.380 | 10.901 | 0.293 | 1.027 (n=83) | 0.872 (n=8) | 5/5 | PASS |
| CHM00054527 | TIANJIN | 257.8 | 3029 | 81.68% | 45.35% | 64.69% | 2.443 | 11.249 | 0.215 | 0.754 (n=83) | 0.885 (n=8) | 5/5 | PASS |
| CHM00058027 | XUZHOU | 288.0 | 3063 | 74.40% | 35.35% | 73.16% | 3.112 | 13.047 | 0.206 | 1.500 (n=83) | 1.469 (n=8) | 5/5 | FAIL |

选中 `CHM00054823`（JINAN），距 YC 49.85 km。五个目标日 NOAA source flag 为 `s,s,s,s,s`，quality flag 为 `,,,,`。湿日 precision 52.91%，recall 94.98%，日降水 MAE 1.693 mm，官方湿日雨量 MAE 8.252 mm，Pearson r 0.6511。月/年总量比只汇总覆盖率 ≥90% 的期间：完整月 n=83，中位数/范围 1.219 / 0.130–9.217；完整年 n=8，中位数/范围 1.256 / 0.857–1.617。2013 年配对覆盖 117/365 日 (32.1%)，已从完整年比值摘要中排除。湿日 precision 52.91%，表明不匹配湿日仍不少；本站结果仅用于本次五个目标日的限定填补，不代表可无条件替代 YC 长期日降雨序列。完整月份差异示例：2011-06 站/官方 14.9/114.9 mm；2010-02 21.2/2.3 mm。配对来源标志计数：s=2922，S=117。 NOAA 定义 source flag `s` 为中国气象部门来源；`S` 为由全球同步报文汇总，降水需谨慎解释。[GHCN-Daily 官方格式及标志说明](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)

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
- 按 3653 个训练天气日计的来源填补比率：

| 变量 | 已接受 gap-fill 天 | 全训练期占比 |
|---|---:|---:|
| TMAX | 107 | 2.93% |
| TMIN | 107 | 2.93% |
| SRAD | 151 | 4.13% |
| RAIN | 5 | 0.14% |

- 来源层级：2004 非缺失日使用 ChinaFLUX 半小时产品（TMAX/TMIN consistency-based，SRAD 积分，RAIN 完整 48 条求和）；目标五天使用通过验证的邻近 GHCN 日雨量。2005–2013 完整官方 QC 值优先，NASA 只填 TMAX/TMIN/SRAD 的已验证缺口；RAIN 保持官方产品值。
- 2004 candidate 年降雨：846.2 mm。ChinaFLUX 年尺度产品参照为 846.2 mm；仅作独立年值参照，不用于分配目标五天。
- Candidate 与 ChinaFLUX 年/月产品比较：年值差 0.0 mm；月值逐项相等 12/12 月（只作一致性对照，不参与填补决策）。详见 `chinaflux_2004_candidate_comparison.csv`。
- Candidate QC：`passed=True`，`not_run=False`；原因：无。
- `yc_wgen_fitting_weather_2004_2013.csv`：已生成 3653/3653 日；日期、变量完整性、物理 Gate 和 provenance QC `True`；年/月统计与 ChinaFLUX 聚合比较见对应 CSV。
- NASA 三变量接受 365 个 gap variable-days；五个雨日由地面站 Gate 解决，合计 gap-fill 为 370/370 个已识别缺口。

## 5. 泄漏与复现

官方源表只解析到 2013；NASA 偏差校正/验证年份为 2005–2013。GHCN `.dly` 只解析 2004-10-16 至 20 及 2005–2013 的 PRCP；2014+ 行仅参与文件 SHA256，未解析天气值。泄漏审计：`True`，最大用于值/校准年份 2013。

原始来源 URL、检索时间、HTTP 元数据和文件哈希见 `run_manifest.json`；仅把分析期间的逐日观测子集保存于 `rain_station_source_values.csv`，没有缓存/提交全站历史数据。

## 6. 下一步

当前状态 `PASS_YC_WEATHER_CANDIDATE`。下一项最小任务是冻结并审阅本轮 candidate，在独立任务中准备 train-only `.CLI` 并开展受控 WGEN pilot；沿用当前单站、训练期范围和资源门槛。本轮未生成 `.CLI`，未运行 WeatherMan/WGEN、DSSAT 或 PPO。
