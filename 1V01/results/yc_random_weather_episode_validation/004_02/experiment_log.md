# YC random-weather episode ensemble 实验记录

## 实验边界

- 任务：验证 YC 的 DSSAT WGEN episode-level 随机天气是否适用于后续受控 PPO pilot。
- 生成范围：仅 YC，weather_seed 1001–1100；同一冻结 CNYC.CLI 和 treatment 1 作物/土壤/品种/管理设置；random_weather=True、WTHER=W。
- 对照：fitting 为 2004–2013 冻结天气；2014–2023 WTH 只作比较，不参与 WGEN fitting、参数调整或 seed 筛选。
- 明确未执行：PPO、WGEN 重拟合、DSSAT/runtime/CLI 修改、完整日历年 WGEN 导出、PPT。
- 资源控制：DSSAT episode 串行运行；单任务容器；TMPDIR 限定在本任务结果目录；不并发占用多份模型/运行内存。

## 运行记录

1. 预检冻结输入 hash：CNYC.CLI `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`；fitting weather `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`。
2. Smoke：seed 1001，DSSAT/WGEN PASS，116 天，天气结束 2008-09-24；日期连续、无重复、天气有限且物理范围检查通过。
3. 批次生成：seed 1002–1100 串行完成，99/99 PASS。连同 smoke 共 100/100 PASS；完整 episode 长度 106–131 天。未删除、补抽或筛选任何 seed。
4. 固定 seed 重跑：1001、1025、1050、1075、1100 均 PASS；五组逐日日期和 RAIN/SRAD/TMAX/TMIN 与首轮完全一致。
5. 分析窗：最短 episode seed 1075 于 2008-09-14 结束，所以统一窗为 2008-06-01 至 2008-09-14，共 106 天。2004–2013 和 2014–2023 各按相同月日提取完整窗口；月降雨分项与窗口总量逐序列闭合。
6. 分析自测中遇到并修复的问题：浮点相关断言改为容差比较；观测年份窗口改为按月日映射至各自年份；episode 长度图改读取完整 season 表。以上是分析代码问题，不涉及重跑 DSSAT。

## 主要结果

- Fixed-window 总雨量均值：fitting 463.54 mm；2014–2023 comparison 398.17 mm；synthetic 494.22 mm。synthetic 相对 fitting 的均值偏差为 0.22 个 fitting 年际 SD，月雨量型相关 r=0.976。
- Synthetic 湿日均值 28.70 天，fitting 观察年均值 29.00 天；湿日阈值保持 RAIN > 0.0 mm。
- TMAX seasonal mean 偏差 +0.45 C（0.56 个 fitting 年际 SD）；TMIN +0.58 C（1.08 个 fitting 年际 SD）。日内分布标准差比接近 1；故 temperature gate 为 PASS_WITH_NOTES，记录轻度偏暖，不据此调参。
- SRAD seasonal mean 偏差为 0.37 个 fitting 年际 SD，日 SD 比值 0.91。
- 100 个全 season weather 序列哈希均不同；五个预指定 same-seed 重跑均逐日一致。
- 各变量分布、分位数、逐月降雨/SRAD、跨变量相关、lag-1 相关与固定窗/完整季统计见同目录 CSV 与 figures。

## 判定与限制

- Generation、reproducibility、diversity、rainfall、SRAD、dependence structure：PASS。
- Temperature：PASS_WITH_NOTES（TMIN 均值轻度偏暖）。总体 `PASS_WITH_NOTES`；`yc_random_weather_ready_for_ppo_pilot=YES`，建议下一轮仍按 smoke-before-formal、单站点与预先固定预算执行。
- 观察年每组只有 10 年；2014–2023 数据是独立于本轮 WGEN fitting 的 comparison，但既有项目审计没有认证其为全项目范围的 pristine holdout。结果是 episode 生长季描述性验证，不是全年、未来气候或极端气候验证。
- 未验证自动全年 WGEN 导出仍为 deferred/optional，不阻挡 episode-level random-weather PPO 工作流。

## 产物

- `episode_generation_qc.csv`
- `fixed_window_weather_by_seed.csv`
- `full_crop_season_weather_by_seed.csv`
- `rainfall_validation_summary.csv`
- `monthly_rainfall_validation.csv`
- `temperature_validation_summary.csv`
- `srad_validation_summary.csv`
- `cross_correlation_validation.csv`
- `serial_correlation_validation.csv`
- `weather_seed_reproducibility.csv`
- `weather_seed_diversity.csv`
- `validation/observed_wth_source_manifest.csv`
- `summary.json`
- 中文综合报告：`docs/yc_random_weather_episode_ensemble_validation.md`

提交：报告、分析代码、汇总表/图、100 个逐 seed episode CSV 和 5 个重跑 CSV 本地提交；重复的逐运行状态/日志/runtime snapshots 留在本地且由本任务目录 `.gitignore` 排除。Git push 未执行，具体提交号由最终终端摘要记录。
