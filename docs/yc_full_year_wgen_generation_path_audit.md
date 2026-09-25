# YC 完整全年 WGEN 生成路径审计

## 结论

```text
NO_VERIFIED_AUTOMATED_FULL_YEAR_WGEN_PATH
selected_path: NONE_VERIFIED
single_full_year_smoke_status: NOT_RUN
```

WeatherMan 是当前最接近目标的官方路径：DSSAT 官方旧版 WeatherMan 手册记载可用 WGEN 生成完整年份、指定随机种子并导出日天气。但它不能证明当前机器已安装可用的 WeatherMan，也不能证明当前版本有命令行/批处理接口或能直接读取正式 YC `CNYC.CLI`。按照项目 `AGENTS.md`，本轮只读项目目录，没有探测项目外安装目录、系统 PATH/注册表或 live Docker。因此不能把“未知”写成“未安装”，也不能据此启动 smoke。

## 1. Blocker 与已知行为

Gate A 已完成且本轮未重拟合参数：正式 CLI 的 168 个参数月值审计状态沿用此前记录；冻结合成 CLI SHA256 为 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`，拟合天气 SHA256 为 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`，与任务说明一致。

Gate B 的问题不是 WGEN 不能逐日生成，而是现有 Gym-DSSAT CSM 作物模拟仅运行到约 116–120 天的单季终点。DSSAT v4.8.0.24 的 CSM 日循环由 `YREND` 控制，源码注释把它定义为季节结束日期、通常为收获日期。因此这段长度反映作物季节边界，而不是 WGEN 的固定 120 天限制。`WEATHR` 在每日 RATE 阶段调用 WGEN；`Weather.OUT` 是 CSM 已模拟日期的逐日输出，不能自动延展至日历年末。完整证据链见 [source_trace.md](../results/yc_wgen_cli_pilot/004_01/full_year_path_audit/source_trace.md)。

## 2. WeatherMan 本机与自动化审计

| 问题 | 本轮结论 |
|---|---|
| WeatherMan 是否安装 | 未知。仓库内 `opt/`、`local_packages/` 未找到匹配文件；项目外路径按边界未检查。 |
| 版本、路径、文件哈希 | 未知；没有可验证的本机可执行文件。 |
| GUI-only / 命令行 / batch | 当前安装状态与接口未知。旧版手册描述菜单式操作；未找到适用于当前 4.8.x 的官方 batch/CLI 契约。 |
| 随机种子 | 旧版官方手册记录可以指定随机数种子以重现序列；当前安装版本尚未验证。 |
| 完整 365/366 天 | 旧版官方手册明确描述按完整年份生成；当前版本/本机尚未验证。 |
| 导出逐日 RAIN/SRAD/TMAX/TMIN | 官方资料记录生成日天气并导出；当前版具体目标格式（WTH/CSV/TXT）和本机能力尚未验证。 |
| 直接使用 `CNYC.CLI` | 未证实。DSSAT CSM 的 WGEN 源码确实从其输入气候文件读取 `*WGEN` 参数，但这不等于 WeatherMan GUI 可把该 CLI 作为生成输入或直接将其转成全年 WTH。 |

本机审计边界和逐项证据见 [local_installation_audit.txt](../results/yc_wgen_cli_pilot/004_01/full_year_path_audit/local_installation_audit.txt)。官方 DSSAT FAQ 介绍 WeatherMan 可依据气候数据生成和导出逐日天气，同时也说明 CSM 内部可选 WGEN/SIMMETEO；旧版完整年份与 seed 证据来自 DSSAT User's Guide Vol. 3。后者为 DSSAT v3 文档，不能外推为当前 4.8.x 的本机能力。[官方 FAQ](https://dssat.net/5165/)，[官方 WeatherMan 参考手册](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf)

## 3. 独立官方 utility 与 CSM 路径

### 独立 utility

项目本地 `opt/` 和 `local_packages/` 文件名审计没有发现 WeatherMan/WGEN 可执行文件。官方 v4.8.0.24 `WGEN.for` 证明 WGEN 是由天气模块逐日调用的 Fortran 子程序，提供 seed 输入，但没有发现本项目可用的独立官方批量生成程序。全局 DSSAT 安装没有检查，因此该路径状态仍为未知，不作不存在的断言。

### CSM 模拟配置

官方调用链为 `CSM DAY_LOOP → LAND → WEATHR(RATE) → WGEN`。CSM 的 `DAY_LOOP` 继续条件是 `YRDOY > YREND`，`YREND` 由 season/land 模块的运行流程传递；注释称通常是收获日期。超过该日期后进入 `SEASEND`，所有 run 完成后执行 `ENDRUN`。源码没有显示一个脱离 crop/land lifecycle 的 WGEN-only 全年运行方式。Seasonal/Sequence 模式存在，但本轮没有证据证明它们会让当前玉米日循环无作物终止并跑至 12 月 31 日；fallow/no-crop 全年配置也未验证。

所以回答“CSM 是否已找到不依赖 crop maturity 的全年 WGEN 配置”：**没有找到已验证配置**。这不是证明通过其他配置绝对做不到，而是当前证据不足以将其作为稳定生成路径。源码路径与版本依据见 [source_trace.md](../results/yc_wgen_cli_pilot/004_01/full_year_path_audit/source_trace.md)。

## 4. 候选路径与选择

候选明细见 [full_year_wgen_path_candidates.csv](../results/yc_wgen_cli_pilot/004_01/full_year_path_audit/full_year_wgen_path_candidates.csv)。WeatherMan GUI 是官方、支持完整年份生成的概念路径，但缺少本机可用性、当前版本自动化、当前版 seed 控制、导出格式及 `CNYC.CLI` 直读的验证，故标为 `VIABLE_WITH_NOTE`，不是已满足自动化门槛的选择。官方 standalone utility 和 CSM no-crop 全年路线仍为 `UNKNOWN`。

**正式建议：目前不启动全年批量生成，也不修改 Gym-DSSAT/DSSAT。** 下一步先做一次人工 WeatherMan 单年 smoke：确认可执行程序版本、显式选择 WGEN、确认参数来源和 seed 输入、生成 1 个完整年并导出，再核实 365/366 行和 QC。GUI 人工步骤未执行；不得把其结果记录成已完成。若本机无 WeatherMan，再调查隔离的官方 utility；只有此前路径都排除后才评估官方 WGEN 源码的隔离包装。

## 5. Smoke 与环境改动

- 单年 smoke：未运行；无行数、开始日期、结束日期或 QC 结果。
- 不生成 100 年；不运行 PPO。
- 当前 Gym-DSSAT/DSSAT runtime：未修改。
- `CNYC.CLI`：未修改，SHA256 与任务输入一致。
- 冻结拟合天气：未修改，SHA256 与任务输入一致。
- WGEN 参数：未重拟合；湿日定义未变更。
- 未安装软件、重建 runtime、修改二进制或创建 PPT。

## 6. 文件、验证与 Git

本任务新增：

- `results/yc_wgen_cli_pilot/004_01/full_year_path_audit/full_year_wgen_path_candidates.csv`
- `results/yc_wgen_cli_pilot/004_01/full_year_path_audit/source_trace.md`
- `results/yc_wgen_cli_pilot/004_01/full_year_path_audit/local_installation_audit.txt`
- `results/yc_wgen_cli_pilot/004_01/full_year_path_audit/path_decision.json`
- `docs/yc_full_year_wgen_generation_path_audit.md`

验证计划为检查 JSON 可解析、CSV 列/候选行完整、`git diff --check` 通过，并复核两个冻结输入哈希。本任务未新增脚本，因此不需要额外单元测试。Git 只提交上述五个任务产物；其他既有工作区修改不纳入，且不推送远端。

最终机器可读判定见 [path_decision.json](../results/yc_wgen_cli_pilot/004_01/full_year_path_audit/path_decision.json)。
