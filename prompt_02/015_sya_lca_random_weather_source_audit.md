# 015 SYA / LCA 随机天气增强前置天气来源审计

## 任务目标

对 SYA（沈阳）和 LCA（栾城）进行随机天气增强前置审计。

目标：

在不改变 reward、action mask、动作空间、IC 配置和 PPO 设置的情况下，仅判断两个站点是否具备按照 YC 相同流程进入 WGEN 随机天气增强的条件。

本任务只做天气来源审计，不运行 WGEN、DSSAT、PPO。

---

## 1. 当前配置核查

读取当前 SYA/LCA baseline 配置，确认：

- site
- IC类型
- reward function
- action space
- action mask
- irrigation limit
- nitrogen limit
- training years
- validation years
- weather source

输出：

`baseline_config_summary.csv`

字段：

- site
- config_file
- IC
- reward
- action_space
- mask_version
- train_weather_source
- train_year_range
- validation_weather_source
- validation_year_range

禁止修改任何配置。

---

## 2. 天气来源完整性审计

分别检查 SYA 和 LCA：

- 当前训练天气文件位置；
- 使用年份范围；
- WTH 文件是否完整；
- 日期是否连续；
- 是否包含：
  - RAIN
  - SRAD
  - TMAX
  - TMIN

检查：

- 缺失日期；
- 重复日期；
- 非法值；
- 降水负值；
- 辐射负值；
- 温度逻辑错误。

输出：

`weather_source_audit.csv`

字段：

- site
- year
- file
- days_expected
- days_found
- missing_days
- duplicate_days
- rain_missing
- srad_missing
- tmax_missing
- tmin_missing
- qc_status

---

## 3. WGEN兼容性判断

判断 SYA/LCA 是否可以直接进入 YC 同样的随机天气流程。

确认是否可以形成连续逐日天气：

- RAIN
- SRAD
- TMAX
- TMIN

并判断是否可以利用训练期历史天气：

- 计算月统计参数；
- 生成对应站点 CLI；
- 进入 WGEN。

---

## 4. 缺口处理原则

如果发现天气缺口：

不要自动补。

只报告：

- 日期；
- 变量；
- 缺失数量；
- 当前来源；
- 可能原因。

分类：

- COMPLETE_READY_FOR_WGEN
- BLOCKED_BY_DATA_PROVENANCE

---

## 5. 禁止事项

禁止：

1. 修改天气文件；
2. 自动填补缺失；
3. 使用 NASA POWER；
4. 使用邻站数据；
5. 生成 CLI；
6. 运行 WGEN；
7. 运行 DSSAT；
8. 运行 PPO。

---

## 6. 输出文件

生成：

`docs/sy_lc_random_weather_source_audit_015.md`

生成：

`results/sy_lc_random_weather_015/`

包含：

- baseline_config_summary.csv
- weather_source_audit.csv
- final_gate.json

final_gate.json：

{
"SYA_weather_ready":"",
"LCA_weather_ready":"",
"need_gap_fill":"",
"next_step":""
}

---

## 7. Git

不提交原始天气文件。

可提交：

- prompt
- audit script
- csv
- md

建议 commit：

`sya_lca: audit weather source readiness for random weather enhancement`

未经确认不要 push。
