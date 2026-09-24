# YC `CNYC.CLI` 可复现生成器实现记录

## 1. 任务范围

本轮只把 YC 训练期天气统计拟合成 DSSAT/WGEN 格式 candidate，并做静态检查。没有运行 DSSAT、WGEN、Gym-DSSAT 或 PPO，也没有改动天气输入、PPO 配置、其他站点或冻结基线。

生成状态：`CANDIDATE_READY_FOR_WGEN_SMOKE`。这是静态 candidate，不代表已通过运行时 smoke。

## 2. 冻结输入

- 输入：`results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv`
- SHA256：`4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`，生成前核验一致
- 日期：2004-01-01 至 2013-12-31，共 3653 个连续日，含 2004、2008、2012 闰日
- 输入列：`DATE, SRAD, TMAX, TMIN, RAIN`
- 拟合只读取上述文件；没有读取或使用 2014–2023 验证天气

生成器默认强制 SHA256、日期范围、行数、唯一日期和逐日连续检查。它拒绝缺失/非有限气象值、`TMAX<TMIN`、负降水及负辐射。输入以只读方式处理。

## 3. 定义依据与适用边界

字段含义及区段顺序参照 DSSAT User's Guide Volume 3 WeatherMan Appendix B；`START/DURN`、`GSST/GSDU` 的字段含义同时核对 Volume 2 weather data codes。当前仓库 `CNSY.CLI` 只用于标签、顺序、宽度和格式结构，不复制其任何气候数值。

核心拟合采用 Richardson 与 Wright 原始 WGEN PAR 附录中的可执行算法：按降雨状态拆分 Tmax/SRAD，Tmin 合并全部日期；标准差使用样本分母 `n-1`；降雨转移概率按目标日月份记数；湿日 Gamma 形状参数按源程序中的 Greenwood–Durand 近似计算。Soltani 与 Hoogenboom 对 DSSAT WeatherMan 中的 WGEN 说明：WGEN 使用日数据拟合，参数按 calendar month 估计，日参数在生成时内部线性插值。DSSAT 公共 `WGEN.for` 显示运行时由 `RNUM` 得到月湿日概率、由 `RTOT/RNUM` 得到湿日均雨量，再结合 `ALPHA` 得到 Gamma scale。

需保留一项来源差异：WGEN 原始文字说明用 `0.01 inch` 描述湿日，而其参数拟合源码 `PPRAIN` 实际以 `RAIN>0.00` 分组；`MSD` 的 Tmax/SRAD 条件分组也按正值/零值分开。本实现按拟合代码执行 `RAIN>0.0 mm`，而不是将英制阈值机械换算为毫米。冻结 YC 输入中有 48 个正值雨量低于 0.254 mm，因此这项选择会改变统计量，已在字段字典和元数据中明示。

WeatherMan 的 DSSAT 适配按自然月拟合。本实现保留 2 月 29 日并按 2 月归档；原始 WGEN PAR 的 365 日 Fortran 输入程序会跳过 2 月 29 日。两者差异在字典中记录，当前依据是 DSSAT WeatherMan calendar-month 适配说明，而不是把旧程序 365 日索引硬套到现代日期序列。

## 4. 字段字典

机器可读定义见 `results/yc_wgen_cli_pilot/003_06_04/definitions/cli_parameter_definitions.json`。WGEN 每月 14 个必需统计字段全部定义并实现，共 168 个月度数值：

| 字段 | 含义与计算 |
|---|---|
| `SDMN/SDSD` | 干日 SRAD 均值 / 样本标准差 |
| `SWMN/SWSD` | 湿日 SRAD 均值 / 样本标准差 |
| `XDMN/XDSD` | 干日 TMAX 均值 / 样本标准差 |
| `XWMN/XWSD` | 湿日 TMAX 均值 / 样本标准差 |
| `NAMN/NASD` | 不分干湿日的 TMIN 均值 / 样本标准差；原 WGEN PAR 的 TMIN 分支跳过降雨条件筛选 |
| `ALPHA` | 正降雨湿日 Gamma 形状参数；逐月用原 WGEN PAR Greenwood–Durand 多项式估计，`alpha>=1` 时按源代码置为 `0.998` |
| `RTOT/RNUM` | 跨训练年平均月降水总量 / 平均月湿日数；湿日判定为 `RAIN>0` |
| `PDW` | `dry_to_wet / (dry_to_dry + dry_to_wet)`；转移记入当前日所在月 |

`P(wet|wet)` 不单独存入此 CLI 列表；按 DSSAT WGEN 读取逻辑由 `PW=RNUM/当月天数` 和 `PDW` 推导，并在独立核对表中审计。转移跨月、跨年连续，`Dec 31 -> Jan 1` 计入 1 月；首条观测按原参数程序的 `RIM1=0` 初始化为“前一日干”。

## 5. 月统计与气候元数据

- `*MONTHLY AVERAGES`：按训练期自然月计算日均 SRAD/TMAX/TMIN；`RTOT` 与 `RNUM` 为 10 个同月年值的算术平均。
- `TAV`：训练期日均温 `(TMAX+TMIN)/2` 的平均值。
- `AMP`：训练期 12 个自然月平均日均温中，最暖与最冷月均值差的一半。
- `SRAY/TMXY/TMNY`：训练期日均 SRAD/TMAX/TMIN 的平均值。
- `RAIY`：逐年 RAIN 总量的 10 年平均值。
- `START=2004`、`DURN=10`：气候汇总窗口起始年与年数。
- `INSI/LAT/LONG/ELEV`：经核实的 YC 元数据 `CNYC, 36.830, 116.570, 22 m`。

以下数据缺乏 YC 依据，且不是 WGEN 所需统计参数：`ANGA/ANGB`、`AMTH/BMTH`、`SHMN`、`REFHT/WNDHT`、`GSST/GSDU`。它们以 `-99` 缺失哨兵写入，不借用参考期未知的旧值或其他站点参数。`GSST/GSDU` 是生长季起始日/持续天数，不是 WGEN 参数。范围检查阈值为 WeatherMan 可编辑的站点归档 QC 元数据，本任务没有 YC 阈值来源，所以 `MIN/MAX/RATE` 写为 `-99`；Above/Below/Rate 计数也写 `-99` 表示未评估，而不是声称零异常。该缺失只影响归档 QC 元数据，不影响 WGEN 系数拟合。

## 6. 实现与输出

脚本：`scripts/build_dssat_cli.py`。实现仅依赖 Python 标准库；重复输入和代码会生成字节一致的 `CNYC.CLI`。`.CLI` 本体不含时间戳，生成时刻保存在 JSON 元数据。

输出目录：`results/yc_wgen_cli_pilot/003_06_04/`

- `definitions/cli_parameter_definitions.json`
- `generated/CNYC.CLI`
- `generated/monthly_wgen_statistics.csv`
- `generated/monthly_weather_summary.csv`
- `generated/cli_generation_metadata.json`
- `generated/cli_generation_log.txt`
- `validation/cli_schema_check.json`
- `validation/parameter_crosscheck.json`
- `validation/test_results.txt`
- `experiment_log.md`

生成的站点气候摘要为 `TAV=13.9°C`、`AMP=14.8°C`、`SRAY=13.5 MJ m-2 d-1`、`TMXY=19.5°C`、`TMNY=8.3°C`、`RAIY=645 mm year-1`。气候值均由冻结训练期数据重新计算。

## 7. 测试和独立交叉核对

`python -m unittest tests/test_build_dssat_cli.py -v`：9 项测试全部通过。覆盖冻结哈希及错哈希拒绝、月分组、湿干分类、跨年转移、参数计算、闰日、`TMAX<TMIN` 拒绝、负 RAIN/SRAD 拒绝、确定性输出和 12 月完整性。

独立路径使用 Python `statistics.mean/stdev` 对月条件均值/样本标准差复算，并重新计数 PDW、`P(wet|wet)`、RTOT、RNUM，另行复算 ALPHA。12 个月、每月 15 个量，共 180 项核对通过；无任何差异。静态 schema 检查确认五个区段齐全、两组月份均为 1–12、数值可解析且无 NaN/Inf。

## 8. Candidate 与未运行事项

- `CNYC.CLI` SHA256：`5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`
- `cli_generated=YES`
- `cli_status=CANDIDATE_READY_FOR_WGEN_SMOKE`
- 未运行 WGEN/DSSAT/Gym-DSSAT/PPO；没有任何生成天气或训练结果可报告。
- 非 WGEN 站点元数据和归档范围阈值仍用明确的 `-99` 缺失哨兵；下一阶段需先验证运行时是否接受该 candidate 的这些可选字段。

建议的隔离 seed pilot：固定 YC crop/soil/management 和模型配置，仅将 `CNYC.CLI` 配入 `random_weather=True`，依次使用 `weather_seed=101,102,103,104,105`。先检查同 seed 重跑天气逐日完全一致、不同 seed 的天气确有变化，再决定是否进入 DSSAT smoke；本轮未执行这些步骤，也不扩展到 PPO。

## 9. 来源

1. DSSAT User's Guide Vol. 3, WeatherMan Reference Guide, Appendix B: <https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf>
2. DSSAT User's Guide Vol. 2, Weather Data Codes: <https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol2.pdf>
3. Richardson & Wright (1984), *WGEN: A Model for Generating Daily Weather Variables*, Appendix D, WGEN PAR source listing: <https://support.goldsim.com/hc/en-us/article_attachments/115026531468>
4. Soltani & Hoogenboom (2003), “A statistical comparison of the stochastic weather generators WGEN and SIMMETEO,” *Climate Research* 24:215–230, doi: <https://doi.org/10.3354/cr024215>
5. DSSAT public WGEN source, `Weather/WGEN.for`: <https://github.com/DSSAT/dssat-csm-os/blob/develop/Weather/WGEN.for>

## 10. 版本控制记录

任务开始时工作区存在其他文件的既有修改与未跟踪产物。本轮未清理、暂存或改写这些文件；仅会显式暂存上述任务文件。GitHub backup 状态：`pending explicit user approval`；未执行 `git push`。
