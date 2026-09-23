# YC 训练期 WGEN 天气链路核验记录（003_02）

- 执行日期：2026-09-23（Asia/Shanghai）
- 最终状态：`BLOCKED_WEATHER_INTEGRITY`
- 范围：仅 YC/YCA；未生成 `.CLI`、未生成随机天气、未运行 DSSAT smoke、未启动 PPO。

## 1. 执行摘要

Gate A 的站点标识和文件名前缀静态映射已核实：低 IC FileX 的站点字段为 `CNYC0801`/`CNYC1401`，safe renderer 按年份重写为 `CNYCyy01`，并复制同名 `.WTH`；DSSAT 气候文件名使用四字符站点码，对应预期 `CNYC.CLI`。当前训练参数仍为 measured-weather 模式（`random_weather=false`），本轮没有运行 WGEN，所以没有把静态映射夸大成运行时验证。

Gate B 的 `.WTH` 格式检查 10/10 通过：每年完整覆盖 365/366 天，无缺日、重复日、解析错误、已知 sentinel、负降雨/负辐射或 `TMAX<TMIN`；全部逐日、逐变量与清洗表在 0.051 单位舍入容差内匹配。但清洗前源数据表显示 2004–2013 每一年均有 SRAD 或温度缺值，且已经用站点-月份均值填补。尤其 2004 年原始温度各缺 47 天、辐射缺 41 天；原始降雨表 366 天中 348 天为空，最长 257 天无雨段与一段连续原始空白完全重合，后经预处理转成 0。故 2004 的 `67.4 mm / 257 d` 不是已核实的完整观测事实；以当前证据拟合 WGEN 会把未解决的源缺测当成气候信号。

**结论：** 按任务规则在 Gate B 停止。未用验证年份或旧 `CNYC.CLI`；未生成、修补或覆盖任何 `.CLI`/`.WTH`；未执行 seed、pilot 天气或 DSSAT/PPO。

## 2. 研究边界与执行保护

- 仅检查 YC/YCA 与 2004–2013；validation 2014–2023 不参与天气参数计算。
- `configs/055_00_yca_lowIC_expanded_action_maskableppo.json:24-25` 定义 train/validation 年份；独立 test 仍为 `not_available_or_not_verified`。
- 原始 Excel 通过 Excel COM 以只读方式打开；DSSAT 原始输入只读；未改 LC、SY、HL、FQ 或 PPO 生产代码/配置。
- 执行开始时 HEAD=`c0c1f9f`，分支=`codex/sya-forecast-freeze-2026-08-16`。当时有 5 个无关的已修改跟踪文件；本轮不纳入、不覆盖、不提交这些改动。既有未跟踪研究/构建目录也未清理。

## 3. Gate A：FileX / WSTA 映射

证据链：

1. 当前 `055_00` 的 `input_profile=lowIC` 指向 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`（`src/run_sya_ppo_configured_046_02.py:31-33`）；runner 预检读取该目录下 `YC/CNYC0801.MZX`（`src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py:89-95`）。
2. `CNYC0801.MZX:21-23` 的 `WSTA` 样例为 `CNYC0801` 和 `CNYC1401`。`src/ppo_safe_rendering.py:42-48,77-91,258-289` 将年份相关字段渲染为 `CNYCyy01`；`src/ppo_safe_rendering.py:300-304` 将相应 `CNYCyy01.WTH` 复制到运行目录。当前 runner 传入 `experiment_number=1` 且 `random_weather=false`（`src/ppo_safe_rendering.py:317-341`）。
3. 对应历史天气首部为 `$WEATHER DATA : CNYC`，`@ INSI` 值为 `CNYC`（例如 `CNYC0401.WTH:1-5`）。Gym-DSSAT 参考实现将 `random_weather=false` 设为 measured 模式；为 true 才将天气模式设为 WGEN（`references/dssat_pdi.py:88-92`）。
4. DSSAT 官方 `SECLI` 源码将 `FILEW` 的四字符站点前缀用于 `.CLI` 文件名，因此预期名为 `CNYC.CLI`：[DSSAT SECLI.for](https://github.com/DSSAT/dssat-csm-os/blob/develop/InputModule/SECLI.for)。官方 [WeatherMan / DSSAT Volume 3 指南](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf) 描述了站点 climate 文件与 WGEN 参数的处理。

**判定：** `wsta_mapping.json` 记录的静态站点码/文件名映射已核实；当前 measured 路径没有读取 `.CLI`。本轮未调用 WGEN，因此 `CNYC.CLI` 的运行时查找未执行；项目边界不允许检查仓库外 `/opt/dssat_pdi` 安装目录，故当前安装的 DSSAT 版本也未作断言。

## 4. Gate B：训练天气文件与逐日核对

`train_weather_integrity.csv` 对 2004–2013 十个 `.WTH` 重新扫描。所有文件记录数、起止 DOY、无缺日/重复日和基础物理检查均通过。十年每一个 WTH 日记录都能在 `weather_clean/YCA_weather_cleaned.csv` 找到同日记录；四变量差值均不超过 0.051（对应 `.WTH` 一位小数输出舍入）。这验证了 WTH 的结构与清洗表的一致性，不证明清洗表的值都是原始观测。

| 年份 | WTH 年降雨量 mm | 雨日数 >0.1 mm | 最长连续无雨天 | 原始缺测 SRAD/TMAX/TMIN/RAIN（个值） |
|---:|---:|---:|---:|---:|
| 2004 | 67.4 | 13 | 257 | 41 / 47 / 47 / 0 |
| 2005 | 678.4 | 52 | 47 | 59 / 8 / 8 / 0 |
| 2006 | 381.8 | 45 | 78 | 23 / 17 / 17 / 0 |
| 2007 | 571.7 | 63 | 37 | 3 / 0 / 0 / 0 |
| 2008 | 527.1 | 59 | 40 | 36 / 26 / 26 / 0 |
| 2009 | 802.7 | 63 | 38 | 17 / 14 / 14 / 0 |
| 2010 | 739.9 | 59 | 48 | 4 / 3 / 3 / 0 |
| 2011 | 565.9 | 57 | 33 | 42 / 39 / 39 / 0 |
| 2012 | 575.8 | 40 | 75 | 1 / 1 / 1 / 0 |
| 2013 | 727.3 | 41 | 46 | 2 / 0 / 0 / 0 |

表内“原始缺测”来自 `weather_clean/data_check_by_year_before_fill.csv`。十年均至少有一个原始缺值，SRAD 共 228 个、TMAX 共 155 个、TMIN 共 155 个；降雨预处理前缺值计数为 0，是因为空白已先行转换为 0，而不是源表不存在空白。月降雨、TMAX/TMIN/SRAD 均值及 wet/dry 日降雨均值见 `train_weather_monthly_summary.csv`。日降雨超过 100 mm 的记录仅作为极端值保留并供人工复核，不因超出历史范围自动删除。

## 5. 2004 来源追踪

只读原始表逐站点、逐变量筛选 YCA 2004（每源表 366 个日期、无重复）：

| 来源文件 | 原始列 | 有效数值 | 空白 | 显式零 | 结果 |
|---|---|---:|---:|---:|---|
| `my_data/T2.xls` | 日最大值 / 日最小值 | 各 319 | 各 47 | TMAX 0；TMIN 2 | 两个温度变量都需填补 |
| `my_data/D32.xls` | 总辐射总量 (MJ/m²) | 325 | 41 | 0 | 需填补 |
| `my_data/HLLCYCFQ降雨数据.xls` | `20-20合计(mm)` | 18 | 348 | 4 | 348 个空白由清洗规则转成 0 |

`src/weather_preprocess.py:148-150` 明确对降雨 `NaN` 执行 `fillna(0)`；`src/weather_preprocess.py:260-283` 对各天气变量先用站点-月份均值、再用站点年均值填补。2004 的 41/47/47 个 WTH 值均与填补后的清洗 CSV 在舍入精度内逐日一致。

当前 WTH 的最长干段为 DOY 001–257（257 天）；原始降雨表对应这 257 个日期全部为空。既有清洗报告 `weather_clean/data_check_report.md:17-18,57` 记录过“用户确认雨量空白按无雨处理”的统一规则；但该规则本身不能证明这段连续 257 日空白代表完整观测而非观测覆盖中断。加之温度和辐射缺值已经用均值替代，现有证据不能把 2004 认定为可用于 WGEN 参数估计的完整观测年。未擅自修复、排除或改写该年。

## 6. Test split

`test_split_status.json` 沿用前轮审计：train=2004–2013，validation=2014–2023，独立 test=`not_available_or_not_verified`。2000–2003 仅列为候选历史年份；因人工筛选记录不全且存在覆盖 2000–2023 的 YC null-run，不追认成独立 test。test 状态不阻塞本轮数据技术审计。

## 7. `.CLI` 参数估计与来源

- **没有生成 `.CLI`。** 当前 YC lowIC 输入无可信 train-only `.CLI`。
- 不使用旧文件 `DSSAT_auto_validation/run_CNYC0802_DSSAT480_IC0_null_2000_2023/pdi_smoke_test/input_used_by_pdi/CNYC.CLI`：其 SHA256 为 `58dbe11fdb3af9d34cc25644d8fb965627403778eb92da31a8f91619bd100d51`，记录窗口含 2008–2014，无法证明只由 2004–2013 估计；关联运行又是 `random_weather=false`。
- 项目脚本/参考目录未发现可验证的 WeatherMan 参数估计工具。外部安装目录依仓库 `AGENTS.md` 边界未检查；不据此声称系统全局未安装。
- 由于 Gate B 失败，不准备不可信的导入数据，不进入 WeatherMan/CLI 生成。禁止根据现有缺测后 WTH 猜测或拼装系数。

## 8. WGEN seed contract

没有执行 `weather_generation_seed=101–105`。`seed_reproducibility.json` 明确记录为 `not_attempted_blocked_before_cli`，无生成天气 hash。检查签入参考 wrapper 可见：`random_weather=true` 选择 WGEN 模式并从 Gym 环境 NumPy RNG 抽取 DSSAT `RSEED`；当前 `055_00` 设为 false。wrapper 没有独立的 `weather_generation_seed` 参数接口，因此未来需在隔离 pilot adapter 中分离天气 seed 与 PPO seed，再验证相同 seed 逐日 hash 复现、不同 seed 有可测差异；本轮不改生产训练代码。

## 9. Pilot 天气 QC 与 DSSAT smoke

未生成/捕获任何随机 realization，因此 pilot 分布统计、pilot weather QC、DSSAT smoke 均为 **未执行（前置门未通过）**，而非通过或失败。没有生成 `weather_manifest.csv` 或假造天气 hash。`audit_summary.json` 记录 realization=0、smoke=0、PPO=0。

## 10. 最终 Gate 判定

```text
Gate A  静态站点码/路径映射已核实；CLI 运行时尚未执行
Gate B  失败：2004 来源完整性未解决，且 2004–2013 每年存在原始 SRAD/温度缺值
Gate C  未进入：不生成 .CLI
Gate D–G 未进入：无 seed pilot、随机天气 QC 或 DSSAT smoke
Final   BLOCKED_WEATHER_INTEGRITY
```

这是有意在数据完整性门停止，不是整条 WGEN 路线失败。原始 `.WTH`、Excel、FileX、PPO 代码/配置与历史输出均未更改；本轮 PPO 运行数为 0。

## 11. 下一阶段最小工作

1. 先由 YC 气象数据来源/维护者确认原始降雨表 2004 年的空白编码，尤其 DOY 001–257 是否代表无雨、停测或缺失；需要原站点记录或有出处的补充数据，而不能仅重复通用填零规则。
2. 对 2004–2013 每年 SRAD/TMAX/TMIN 的原始缺测确定可追溯处理原则。若这些年不适合作为 WGEN 拟合数据，需用户明确批准新的拟合年份范围；本轮不自行缩减/替换年份。
3. 数据方案获批并通过 Gate B 后，再在允许访问的范围内确认 WeatherMan/DSSAT 版本和调用方式；使用训练期唯一输入建 `.CLI`，记录所有输入/输出 hash。
4. 然后分离 `weather_generation_seed`，先做 101 同 seed 重复和 102 差异检查，再做 3–5 个 pilot、天气 QC 和至少 3 个 DSSAT 单季 smoke；仍不启动 PPO，除非另行授权。

## 12. 产物索引

- 主机读审计：`results/yc_cli_generation_and_weather_qc/audit_yc_raw_workbooks.ps1`、`weather_2004_raw_excel_audit.json`
- 十年逐日 QC：`results/yc_cli_generation_and_weather_qc/audit_yc_train_weather.ps1`、`train_weather_integrity.csv`、`train_weather_monthly_summary.csv`
- Gate 机器记录：`wsta_mapping.json`、`weather_2004_provenance.json`、`test_split_status.json`、`seed_reproducibility.json`、`audit_summary.json`
- 中文过程日志：`results/yc_cli_generation_and_weather_qc/experiment_log.md`
- 汇报：`docs/yc_cli_generation_and_weather_qc.pptx`

PPTX 共 8 页；8/8 最终渲染页已目视检查。包结构、Artifact Tool 重载和版面检查通过；包含 4 张原生表格与 2 个原生图表，结构/版面警告为 0。最终 SHA256：`b2bcd93ab6ef1281362356a91bd28a9d50ff2736195b13e3d1480b60e1e02454`。未在 PowerPoint 桌面程序中打开验证；记录见 `pptx_validation_summary.json`。

**版本控制：** 本地提交只包含本任务专属报告、脚本和结果；不推送 GitHub。完成后的待批准命令为：

```powershell
git push origin codex/sya-forecast-freeze-2026-08-16
```
