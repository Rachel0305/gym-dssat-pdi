# YC WGEN 参数方法审计与完整随机年份验证

## 结论

本轮分别判定三个层面，不能把工程阻断误报为气候质量失败：

| 层次 | 状态 | 结论 |
|---|---|---|
| 参数方法有效性 | `PASS_WITH_NOTE` | 14 个参数的 168 个未序列化月值独立复算全部一致；湿日阈值在经典文献正文和 WGEN PAR 可执行源码之间存在表述差异，已量化披露。 |
| 全年生成工程有效性 | `BLOCKED_GENERATION_PATH` | 项目安全边界内没有确认可调用的官方完整年 WGEN 入口；未生成 synthetic year。 |
| 合成气候有效性 | `NOT_ASSESSED` | 无完整年合成数据，故历史比较和天气质量验证未运行。 |

当前 `yc_random_weather_data_ready_for_ppo=NO`，不能启动 004_02 PPO。阻断点是全年生成路径，不是合成气候验证失败。

## 1. 研究问题与范围

问题一：`scripts/build_dssat_cli.py` 对 YC 月度 WGEN 参数的计算是否符合 WGEN 参数定义与拟合逻辑？

问题二：由 DSSAT WGEN 基于这些参数生成的全年随机天气，是否合理再现可用 YC 历史气候特征？

本轮只完成问题一，并审查问题二的生成路径。未训练 PPO、调整湿日阈值、修改 runtime/坐标、使用其他站点、重拟合参数或使用 2014-2023 数据。

## 2. 冻结数据

| 输入 | 期间 | 日期/日数 | SHA256 | 用途 |
|---|---|---|---|---|
| `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` | 2004-2013 fitting | 2004-01-01 至 2013-12-31；3653 日 | `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34` | 唯一拟合输入 |
| `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI` | 12 历月 | 冻结候选 | `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929` | 只读序列化核对 |
| 独立观测期 | 2014-2023 | 本轮未读取/汇总 | N/A | 只可独立验证；本轮未进入气候比较 |

输入与 CLI 哈希均符合任务给定值，执行前后未修改。已有 seed pilot 是 116-120 日 crop-season 片段，不是全年气候样本，本轮没有复用。

## 3. WGEN 定义与来源核对

DSSAT User's Guide Vol. 3 Appendix B 将 `SDMN/SDSD`、`SWMN/SWSD` 定义为干/湿日 SRAD 均值和标准差；`XDMN/XDSD`、`XWMN/XWSD` 是干/湿日 TMAX 参数；`NAMN/NASD` 是月 TMIN 参数；`ALPHA`、`RTOT`、`PDW`、`RNUM` 分别是降水 Gamma 形状参数、月降水总量、干日后湿日概率、月平均雨日数。

DSSAT v4.8.0.24 的 `Weather/WGEN.for` 是发生器而非拟合器。其 `WGENIN` 从 `.CLI` 读参数；对非零 `RTOT/RNUM`，用 `PW=RNUM/NUMDAY`、`PWW=1-PDW*(1-PW)/PW` 派生湿日转移，并用 `RWET=RTOT/RNUM` 与 `ALPHA` 得到 `BETA=RWET/ALPHA`。运行时将 `ALPHA` 限制于 `[0.01,0.998]`。本报告不把 `.for` 中读入/派生过程误称为历史天气参数拟合；拟合公式另与 Richardson & Wright (1984) WGEN PAR Appendix D 核对。

### 湿日阈值说明

Richardson & Wright (1984) 正文将湿日写作 `>=0.01 inch`（约 `0.254 mm`）；同一报告 Appendix D 的 WGEN PAR Fortran 源码通过 `RAIN>0.00` 计数并分类。DSSAT v4.8.0.24 `WGEN.for` 读取 `.CLI` 的 `PDW/RNUM/RTOT` 等参数，不在该文件内重新拟合历史降雨发生概率。

当前实现使用 `RAIN>0.0 mm`，与 WGEN PAR 可执行源码的正值判断一致，但不等价于同报告正文的 0.01 英寸定义。冻存拟合期共 587 个正降水日，其中 48 日满足 `0<RAIN<0.254 mm`（占全部 3653 日的 1.31%，占正降水日的 8.18%）。这会影响湿/干分组和 PDW、ALPHA、RNUM 及湿/干条件天气参数；按原始降水求和的 RTOT 不受分类阈值影响。本轮不改阈值、不做敏感性重拟合；该差异是 `PASS_WITH_NOTE` 的方法学保留项。

## 4. 14 参数审计

逐参数定义、公式、条件分组、代码位置、影响和状态见 [逐参数审计 CSV](../results/yc_wgen_cli_pilot/004_01/parameter_audit/wgen_parameter_audit.csv)。独立实现见 [审计脚本](../scripts/audit_yc_wgen_parameters_004_01.py)。

| 参数 | 计算口径 | 状态 |
|---|---|---|
| SDMN / SDSD | 每历月干日 SRAD 均值 / 样本 SD (`n-1`) | `PASS_WITH_NOTE` |
| SWMN / SWSD | 每历月湿日 SRAD 均值 / 样本 SD (`n-1`) | `PASS_WITH_NOTE` |
| XDMN / XDSD | 每历月干日 TMAX 均值 / 样本 SD (`n-1`) | `PASS_WITH_NOTE` |
| XWMN / XWSD | 每历月湿日 TMAX 均值 / 样本 SD (`n-1`) | `PASS_WITH_NOTE` |
| NAMN / NASD | 每历月全部日 TMIN 均值 / 样本 SD (`n-1`)，不拆湿/干 | `PASS` |
| ALPHA | 湿日降水 Greenwood-Durand 近似；`alpha>=1` 时设为 `0.998`；各月至少 3 个湿日 | `PASS_WITH_NOTE` |
| RTOT | 先算逐年逐月原始降水总量，再对该历月 10 年取均值；不依赖湿日分类阈值 | `PASS` |
| PDW | 时间序列转移计数 `P(wet today | previous dry)`；按当前日归月，跨月/年连续，首日前态为 dry | `PASS_WITH_NOTE` |
| RNUM | 每年每月统计 `RAIN>0` 日数，再跨 10 年取均值 | `PASS_WITH_NOTE` |

除湿日阈值的文献注释外，未发现定义或计算差异。闰日保留在二月，依真实日期分组。拟合 ALPHA 范围为 `0.5017-0.998`，没有触发 runtime 的 0.01 下限。

## 5. 独立重算与序列化

审计器独立读取冻存 CSV 并验证连续日期与物理值；均值和样本标准差用 `statistics.mean/stdev`，逐日重数干湿转移，月降水与雨日数按“逐年逐月、再跨年平均”汇总，并独立实现 Greenwood-Durand 公式。

| 核对项 | 结果 |
|---|---:|
| 参数数 × 月数 | 14 × 12 = 168 |
| 独立原始值与 `build_dssat_cli.py` 一致 | 168/168 |
| 冻结 CLI 序列化差异被参数小数位舍入解释 | 168/168 |
| 原始值不一致 / 未解释 CLI 差异 | 0 / 0 |
| 输入 / CLI hash 执行前后保持 | 是 / 是 |

差值和舍入容差见 [独立交叉核对 CSV](../results/yc_wgen_cli_pilot/004_01/parameter_audit/parameter_independent_crosscheck.csv)，摘要见 [parameter_method_summary.json](../results/yc_wgen_cli_pilot/004_01/parameter_audit/parameter_method_summary.json)。旧 `parameter_crosscheck.json` 的 15 项月度统计检查不能替代本次 168 行逐参数长表。

## 6. 全年生成路径门禁

官方 DSSAT v4.8.0.24 源码将 WGEN 实现为 `WEATHR` 调用的每日作物模型子程序。当前仓库没有已确认可脱离作物季、连续写出 1 月 1 日至 12 月 31 日天气的入口。已有 YC WGEN pilot 仅 116-120 天，不适合验证年雨量、全年干湿结构或年序列相关。

此前 WeatherMan 工具审计将安装路径/版本记录为未知；项目 `AGENTS.md` 限制访问项目外路径，因此本轮没有探查外部安装目录，也没有猜测 GUI/CLI 接口。Gate B 因而为 `BLOCKED_GENERATION_PATH`。本轮 synthetic year 数为 0、seed 范围为空；没有生成 `synthetic_years/` 或验证图表。机器记录见 [generation_gate.json](../results/yc_wgen_cli_pilot/004_01/validation/generation_gate.json)。

## 7. 天气验证状态

以下指标均属于 Gate B，状态为 `NOT_RUN`，不是 `PASS` 或天气质量 `FAIL`：

| 指标族 | 状态 | 原因 |
|---|---|---|
| fitting / validation / synthetic 年降雨均值、范围、分位数和 IQR/P10-P90 覆盖比例 | `NOT_RUN` | 无完整年 synthetic ensemble |
| 年/月雨日、湿日强度、最大日雨量、最长干湿连续日 | `NOT_RUN` | 同上 |
| 月降雨和雨日季节性 | `NOT_RUN` | 同上 |
| 年/月 TMAX、TMIN 与极端频率 | `NOT_RUN` | 同上 |
| 年/月 SRAD 分布与极端值 | `NOT_RUN` | 同上 |
| RAIN occurrence-SRAD/TMAX、TMAX-TMIN、TMAX-SRAD 关系 | `NOT_RUN` | 同上 |
| TMAX/TMIN/SRAD/wet-dry lag-1 自相关 | `NOT_RUN` | 同上 |
| 5 seed 同 seed 复现、100 seed 多样性与天气物理 QC | `NOT_RUN` | 同上 |

2014-2023 独立验证期没有参与参数重估、阈值选择、种子筛除或再生成；没有合并 fitting 与 validation。机器摘要见 [synthetic_weather_validation_summary.json](../results/yc_wgen_cli_pilot/004_01/validation/synthetic_weather_validation_summary.json)。

## 8. 适用范围与限制

本研究的设计目标是使用 WGEN 表示历史 YC 气候周围的随机变率；本轮因全年生成门禁阻断，尚未实际生成天气。WGEN 不是未来气候情景生成器，也不是超历史极端事件生成器。本轮没有证据支持“合理再现”或长期气候代表性结论。后续即便 Gate B 通过，结论也仅能限于：合成天气合理再现可用历史记录所代表的 YC 气候统计特征。

## 9. 最终决定与下一步

- `parameter_audit_status=PASS_WITH_NOTE`；湿日阈值不变。
- `full_year_generation_status=BLOCKED_GENERATION_PATH`。
- `synthetic_climate_status=NOT_ASSESSED`。
- `yc_random_weather_data_ready_for_ppo=NO`。
- 下一最小步骤：确认一个可在项目边界内审计调用的官方 WeatherMan/WGEN 全年生成工具及版本/接口证据；先生成一个完整日历年 smoke，检查 365/366 天、日期连续、可复现和物理 QC，再评估是否进入 100 年 ensemble。不得用 crop maturity 截断片段代替全年。

## 10. 测试、文件与 Git

相关回归命令与结果：

```text
python -m pytest -q tests/test_build_dssat_cli.py tests/test_yc_wgen_seed_pilot.py tests/test_yc_wgen_parameter_audit_004_01.py
25 passed, 0 failed
```

新增独立审计测试 2 项。开发阶段首次执行的新审计器曾因 RTOT 聚合类型错误退出，修正后重跑通过；详见实验记录。

本任务新增：

- `scripts/audit_yc_wgen_parameters_004_01.py`
- `tests/test_yc_wgen_parameter_audit_004_01.py`
- `results/yc_wgen_cli_pilot/004_01/parameter_audit/` 下 2 个 CSV、1 个 JSON
- `results/yc_wgen_cli_pilot/004_01/validation/` 下 2 个 JSON
- `results/yc_wgen_cli_pilot/004_01/experiment_log.md`
- 本报告

未生成 synthetic-year CSV、图表或 PPT。提交前工作区另有既存无关修改/未跟踪文件；本轮只提交本任务文件，保留其余内容。提交哈希见最终摘要；未 push。

## 参考文献与官方资料

1. Richardson, C. W. (1981). *Stochastic simulation of daily precipitation, temperature, and solar radiation*. Water Resources Research, 17(1), 182-190. [doi:10.1029/WR017i001p00182](https://doi.org/10.1029/WR017i001p00182)
2. Richardson, C. W., & Wright, D. A. (1984). *WGEN: A model for generating daily weather variables*. USDA ARS-8. [报告与 WGEN PAR Appendix D 源码](https://support.goldsim.com/hc/en-us/article_attachments/115026531468)
3. Soltani, A., Latifi, N., & Nasiri, M. (2000). *Evaluation of WGEN for generating long term weather data for crop simulations*. Agricultural and Forest Meteorology, 102(1), 1-12. [doi:10.1016/S0168-1923(00)00100-3](https://doi.org/10.1016/S0168-1923(00)00100-3)
4. Soltani, A., & Hoogenboom, G. (2007). *Assessing crop management options with crop simulation models based on generated weather data*. Field Crops Research, 103(3), 198-207. [doi:10.1016/j.fcr.2007.06.003](https://doi.org/10.1016/j.fcr.2007.06.003)
5. DSSAT. *User's Guide Vol. 3*, WeatherMan Appendix B, climate file fields. [官方手册 PDF](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf)
6. DSSAT/dssat-csm-os. *Weather/WGEN.for*, tag `v4.8.0.24`. [官方源码](https://github.com/DSSAT/dssat-csm-os/blob/v4.8.0.24/Weather/WGEN.for)
7. Wang, Z., Xiao, S., Wang, J., Parab, A., & Patel, S. (2025). *Reinforcement Learning-Based Agricultural Fertilization and Irrigation Considering N₂O Emissions and Uncertain Climate Variability*. AgriEngineering, 7(8), 252. [doi:10.3390/agriengineering7080252](https://doi.org/10.3390/agriengineering7080252)

---

## 终端摘要

```text
=== YC WGEN PARAMETER AUDIT + SYNTHETIC CLIMATE VALIDATION SUMMARY ===

fitting_weather: results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv
fitting_period: 2004-01-01 through 2013-12-31; 3653 days
fitting_weather_sha256: 4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34
independent_validation_period: 2014-2023; NOT_RUN
parameter_audit_status: PASS_WITH_NOTE
parameters_checked: 14
parameter_month_values_crosschecked: 168/168 raw exact; 168/168 CLI rounding explained
parameter_definition_mismatches: 0 raw mismatch; wet-day prose/code note documented
wet_day_definition: RAIN > 0.0 mm (WGEN PAR code); narrative separately says >=0.01 inch
wet_day_definition_changed: NO
full_year_generation_method: NOT_RUN; no verified accessible official full-year entry point
synthetic_year_count: 0
weather_seed_range: N/A
same_seed_reproducibility: NOT_RUN
different_seed_diversity: NOT_RUN
physical_qc: NOT_RUN
observed_fitting_annual_rain_mean: NOT_COMPUTED
observed_validation_annual_rain_mean: NOT_COMPUTED
synthetic_annual_rain_mean: NOT_COMPUTED
observed_fitting_annual_rain_range: NOT_COMPUTED
observed_validation_annual_rain_range: NOT_COMPUTED
synthetic_annual_rain_range: NOT_COMPUTED
synthetic_within_observed_IQR_pct: NOT_COMPUTED
synthetic_within_observed_P10_P90_pct: NOT_COMPUTED
synthetic_within_observed_min_max_pct: NOT_COMPUTED
annual_rainfall_status: NOT_RUN
rainfall_structure_status: NOT_RUN
temperature_status: NOT_RUN
srad_status: NOT_RUN
cross_correlation_status: NOT_RUN
serial_correlation_status: NOT_RUN
systematic_bias_detected: NOT_ASSESSED
synthetic_climate_status: NOT_ASSESSED_BLOCKED_GENERATION_PATH
ppo_training_run: NO
runtime_environment_modified: NO
wgen_refit_after_validation: NO
validation_data_used_for_fitting: NO
ppt_created: NO
yc_random_weather_data_ready_for_ppo: NO
recommended_next_step: Confirm an official accessible full-year DSSAT WGEN path; run a one-year smoke before any 100-year ensemble.
report_md: docs/yc_wgen_parameter_audit_and_synthetic_weather_validation.md
results_directory: results/yc_wgen_cli_pilot/004_01/
tests_status: 25 passed, 0 failed
git_commit: see final summary
git_push: NO
github_backup_status: NOT_PUSHED
```
