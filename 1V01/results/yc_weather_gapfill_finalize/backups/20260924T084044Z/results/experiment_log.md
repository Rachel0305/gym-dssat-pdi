# 实验记录：YC 变量级天气缺口补齐（003_05_01）

- 执行时间：2026-09-24T08:34:50.031040+00:00
- 分支：codex/sya-forecast-freeze-2026-08-16；开始 HEAD：03111a4341206839e715755266c87f93755c11a2
- 最终状态：`PASS_YC_WEATHER_CANDIDATE`
- 范围：YC/YCA，天气年份 2004–2013；未运行 `.CLI`、WeatherMan、WGEN、DSSAT、PPO。

## Gate A

- TMAX: `PASS`；检查项：`{"TMAX_coverage": true, "TMAX_corrected_MAE": true, "TMAX_pearson_r": true}`
- TMIN: `PASS`；检查项：`{"TMIN_coverage": true, "TMIN_corrected_MAE": true, "TMIN_pearson_r": true}`
- SRAD: `PASS`；检查项：`{"SRAD_coverage": true, "SRAD_corrected_MAE": true, "SRAD_pearson_r": true}`
- RAIN: `BLOCKED`；检查项：`{"RAIN_coverage": true, "RAIN_event_agreement": false, "RAIN_monthly_total_ratio": true, "RAIN_annual_total_ratio": true}`
- RAIN 事件一致率：0.719805；门槛 0.800000，未降低。
- NASA 接受填补：{'TMAX': 107, 'TMIN': 107, 'SRAD': 151, 'RAIN': 0}。只接受 TMAX/TMIN/SRAD，且值与旧表 `raw + correction` 复算一致。

## 本地雨源检查

- ChinaFLUX 日产品有效雨量日：361/366。
- 五天原值：`[{"date": "2004-10-16", "raw_rain_value": -99999, "usable_daily_value": false}, {"date": "2004-10-17", "raw_rain_value": -99999, "usable_daily_value": false}, {"date": "2004-10-18", "raw_rain_value": -99999, "usable_daily_value": false}, {"date": "2004-10-19", "raw_rain_value": -99999, "usable_daily_value": false}, {"date": "2004-10-20", "raw_rain_value": -99999, "usable_daily_value": false}]`；全部缺测。
- 压缩包中的降水及人工气象记录是月统计，不能代替逐日值。

## GHCN-Daily

- 搜索半径：300 km；站点数：12；有效候选站：`['CHM00054823', 'CHM00054725', 'CHM00054618', 'CHM00054916', 'CHM00053898', 'CHM00054909', 'CHM00054843', 'CHM00054527']`。
- CHM00054823 JINAN，49.85 km，配对 3039 日，事件一致率 85.32%，五天 5/5，Gate PASS；原因：[]。
- CHM00054725 HUIMIN，113.28 km，配对 3038 日，事件一致率 86.34%，五天 5/5，Gate PASS；原因：[]。
- CHM00054618 POTOU，139.34 km，配对 3024 日，事件一致率 85.35%，五天 5/5，Gate PASS；原因：[]。
- CHM00054916 YANZHOU，142.67 km，配对 3027 日，事件一致率 84.67%，五天 5/5，Gate PASS；原因：[]。
- CHM00054616 CANGZHOU，168.72 km，配对 0 日，事件一致率 NA（无有效配对），五天 0/5，Gate FAIL；原因：['有效重叠日不足3000', 'rain/no-rain事件一致率低于80%', '2004目标五天有效PRCP仅0/5']。
- CHM00054906 HEZE/CAOZHOU，203.26 km，配对 0 日，事件一致率 NA（无有效配对），五天 0/5，Gate FAIL；原因：['有效重叠日不足3000', 'rain/no-rain事件一致率低于80%', '2004目标五天有效PRCP仅0/5']。
- CHM00053898 ANYANG，212.60 km，配对 3042 日，事件一致率 82.45%，五天 5/5，Gate PASS；原因：[]。
- CHM00054909 DINGTAO，216.14 km，配对 3019 日，事件一致率 82.05%，五天 5/5，Gate PASS；原因：[]。
- CHM00053698 SHIJIAZHUANG，232.44 km，配对 3029 日，事件一致率 79.30%，五天 5/5，Gate FAIL；原因：['rain/no-rain事件一致率低于80%']。
- CHM00054843 WEIFANG，232.76 km，配对 3053 日，事件一致率 81.46%，五天 5/5，Gate PASS；原因：[]。
- CHM00054527 TIANJIN，257.78 km，配对 3029 日，事件一致率 81.68%，五天 5/5，Gate PASS；原因：[]。
- CHM00058027 XUZHOU，288.03 km，配对 3063 日，事件一致率 74.40%，五天 5/5，Gate FAIL；原因：['rain/no-rain事件一致率低于80%']。

## QC / 泄漏 / 决策

- Candidate created: `True`；QC: `{'passed': True, 'not_run': False, 'row_count': 3653, 'expected_days': 3653, 'unique_dates': 3653, 'errors': [], 'physics_failure_count': 0}`。
- 2004 candidate 年降水：`846.2`；ChinaFLUX 年统计参照 846.2 mm（不用于拆分）。
- Leakage audit passed: `True`；weather/calibration max year: 2013。
- PPT 结构校验随后运行；是否可视觉渲染以验证文件为准。
- 下一步：寻找并验证可接受的 2004 五天日降雨；不调阈值，不进入 WGEN/DSSAT/PPO。
