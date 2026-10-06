# SYA WGEN readiness review 任务

## 背景

已完成 SYA train-only fill 敏感性审查与 CLI 参数审计。

当前状态：

- SYA：
  - 已构造仅使用 2005–2013 训练期资料的 train-only fill 候选天气序列；
  - 已确认未使用 2014–2023 验证期资料；
  - CLI 参数审计通过，状态：
    `READY_FOR_WGEN_REVIEW`

- 当前候选输入：
  `results/sya_train_only_fill_sensitivity/candidate_weather_2005_2013.csv`

- 当前已知限制：
  - 2005 年存在集中填补；
  - 462 项 SRAD/TMAX/TMIN 缺口中，424 项位于 2005 年；
  - 均值填补可能降低部分日期尺度日际变率。

本任务目标：
评估该候选天气序列是否适合作为 WGEN 参数拟合输入。

## 禁止事项

本任务只进行 readiness review。

禁止：
1. 运行 WGEN；
2. 生成 synthetic weather；
3. 运行 DSSAT；
4. 运行 PPO；
5. 修改原始 WTH；
6. 修改正式 weather 输入目录；
7. 修改已有 CLI 审计结果。

## 输入

固定：

- 站点：SYA
- 时间范围：2005–2013
- 使用：
  `candidate_weather_2005_2013.csv`

同时保留：
- 原始 WTH；
- 原 provenance audit；
- 018 CLI audit。

记录输入 SHA256。

## 审查内容

### 1. 天气序列统计特征检查

检查候选序列：

- 日尺度连续性；
- SRAD、TMAX、TMIN 分布；
- 月尺度均值；
- 月尺度标准差；
- 极值范围。

重点关注：

2005 年集中填补月份：
- 1月；
- 2月；
- 3月；
- 4月；
- 5月。

比较：
- 训练期整体统计；
- 2005 单年统计；
- 2006–2013 统计。

判断集中填补是否导致异常平滑。

### 2. 填补影响评估

针对 462 个填补位置：

统计：

- 每月填补数量；
- 连续填补长度；
- 填补前后：
  - SRAD 标准差变化；
  - TMAX 标准差变化；
  - TMIN 标准差变化。

重点判断：

- 是否明显降低月内变异；
- 是否影响 WGEN 所需统计特征。

### 3. CLI 与 WGEN 输入适配检查

检查：

- 018 生成的 CLI 参数；
- 月度参数完整性；
- Gamma 参数；
- 降水统计；
- 湿日/干日统计。

确认：
是否存在阻止 WGEN 的参数问题。

## 输出

生成：

`docs/sya_wgen_readiness_review_019.md`

以及：

`results/sya_wgen_readiness_review_019/`

包含：

- `weather_variability_summary.csv`
- `fill_impact_summary.csv`
- `wgen_readiness_gate.json`

## 判定标准

输出之一：

### 通过：
`READY_FOR_WGEN`

### 有条件通过：
`READY_WITH_DOCUMENTED_LIMITATION`

表示：
可以进入 WGEN，但需记录均值填补导致的变率限制。

### 阻塞：
`BLOCKED_BY_WEATHER_STATISTICS`

表示：
候选序列统计特征不适合 WGEN。

不要自行修改判断标准。

## 完成后停止

完成审查后停止，等待下一步授权。
