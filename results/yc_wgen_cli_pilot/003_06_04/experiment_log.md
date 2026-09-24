# 实验记录：YC train-only `CNYC.CLI` 生成（003_06_04）

- 执行日期：2026-09-24（Asia/Shanghai）
- 范围：仅 YC，天气拟合窗口 2004–2013；未运行 DSSAT/WGEN/Gym-DSSAT/PPO。
- 输入：`results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv`
- 冻结 SHA256：`4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`
- 实际输入 SHA256：一致；行数 3653；日期完整连续；验证天气未使用。
- 站点：CNYC，36.830 N，116.570 E，海拔 22 m。

## 方法与重要口径

- 参数字典：`definitions/cli_parameter_definitions.json`；14 个必需 WGEN 统计字段/月，168 个值全部定义并实现。
- Wet/dry：采用 WGEN PAR 拟合代码的 `RAIN>0.0`。原始手册叙述另有 0.01 inch 阈值；训练天气有 48 个正值但小于 0.254 mm，因此差异会改变分类，选择和依据已记录。
- 转移：跨月/跨年连续，转移归入当前日所在月，首观测之前按干日初始化；保留 2 月 29 日并归入自然月 2 月。
- 降雨：`PDW=P(wet|dry)`；ALPHA 使用 WGEN PAR 代码的 Greenwood–Durand 近似；RTOT/RNUM 是跨年平均月总雨量/湿日数。
- Tmax/SRAD：按湿/干日分别计算均值与 `n-1` 样本标准差；Tmin 不按湿干日拆分。
- TAV/AMP/SRAY/TMXY/TMNY/RAIY：全部由冻结训练期重算。
- 非 WGEN 元数据及归档 QC 阈值无 YC 证据，以 `-99` 标记；Above/Below/Rate 未评估，不伪造成零异常。

## 输出与验证

- Candidate：`generated/CNYC.CLI`
- Candidate SHA256：`5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`
- 状态：`CANDIDATE_READY_FOR_WGEN_SMOKE`
- 月度参数表：`generated/monthly_wgen_statistics.csv`
- 月气候摘要：`generated/monthly_weather_summary.csv`
- 元数据与日志：`generated/cli_generation_metadata.json`、`generated/cli_generation_log.txt`
- 静态 schema QC：`validation/cli_schema_check.json`，PASS；5 区段、12 月行、数值格式通过。
- 独立复算：`validation/parameter_crosscheck.json`，12 月×15 项=180 项全部通过。
- 单元测试：9/9 通过；详见 `validation/test_results.txt`。
- 中文说明与幻灯片：`docs/yc_cli_generator_implementation.md`、`docs/yc_cli_generator_implementation.pptx`。

## 决策与 Git

- `cli_generated=YES`；WGEN、DSSAT、PPO 均未运行。下一步仅建议隔离执行 seed 101–105 pilot，先验证相同 seed 天气完全一致、不同 seed 天气有差异。
- 冻结天气未修改；没有复制其他站点参数；没有使用 validation weather。
- GitHub backup：`pending explicit user approval`；`git push`：`NO`。
- Git commit message：`feat: build reproducible YC DSSAT CLI generator`（将在本记录和所有产物完成后提交；精确 commit SHA 在终端交付摘要中报告）。
- 开始时工作区已有其他未提交改动/未跟踪文件；提交范围限定为本任务新文件。
