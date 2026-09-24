# YC `CNYC.CLI` 生成路径确认（003_06_03）

**决策：** `BLOCKED_BY_WEATHERMAN_ACCESS`。另有 `BLOCKED_BY_UNKNOWN_IMPORT_FORMAT` 与 `BLOCKED_BY_UNKNOWN_CLI_FORMAT` 两项待解。已确认 WeatherMan 是优先的官方参数估计路线，但当前设备上的 WeatherMan / DSSAT 版本、CSV 导入规则和 active 4.8.x `.CLI` 严格字段要求均未核实，因此本轮不生成 `CNYC.CLI`。

## 1. 任务范围

本轮只确认冻结 YC 训练天气如何进入官方气候参数生成链，并核对 `.CLI` 格式证据、YC 站点元数据和 Gym-DSSAT 文件传递逻辑。严格不做 WGEN、DSSAT、PPO 运行；不改 PPO、reward、action/observation space、FileX、任何其他站点或生产天气。

## 2. 已冻结输入

| 项目 | 已核实值 |
|---|---|
| 输入 | `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` |
| SHA256 | `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34` |
| 日期范围 | 2004-01-01 至 2013-12-31 |
| 日记录数 | 3653 |
| 主要字段 | `DATE,YEAR,DOY,SRAD,TMAX,TMIN,RAIN` 加来源/QC列 |
| 本轮处理 | 仅核对 hash、表头、行数、起止日期；不重建、不写入、不重新拟合 |

没有使用 2014–2023 validation 日天气。所有气候参数的下一步拟合窗口必须显式限制为 2004/001 至 2013/365。

## 3. 仓库里的 `.CLI` 样本

当前 YC 输入路径没有可用的 `CNYC.CLI`。当前可读样本来自别的站点，仅用于辨认结构，参数值没有复制：

| 样本 | 观察到的结构 | 限制 |
|---|---|---|
| `benchmark_results/027_07_site_specific_stage_maskable_ppo/LC/readiness/baseline_runs/dssat_auto/input/CNLC.CLI` | `*CLIMATE`、站点/位置字段、`GSST/GSDU` | 文件只有 9 行，没有 `*WGEN PARAMETERS`，不能证明 WGEN 完整格式 |
| `benchmark_results/028_05_sy_crossyear_frozen_ppo_daily/2012/seed0/snapshot/CNSY.CLI` | `*CLIMATE`、`*MONTHLY AVERAGES`、`*WGEN PARAMETERS`、`*RANGE CHECK VALUES`、`*FLAGGED DATA COUNT` | 样本的 DSSAT 生成版本未在本轮确认 |

前序 YC 审计记载过一个旧 `CNYC.CLI`（SHA256 `58dbe11fdb3af9d34cc25644d8fb965627403778eb92da31a8f91619bd100d51`），其统计窗口覆盖 2008–2014，不能证明只用 2004–2013 拟合；关联 Gym-DSSAT 运行的 `random_weather=false`。该原始文件在当前记录路径不存在，且不满足 train-only provenance，因此排除，不把前序记录当作当前原始文件。

仓库 `.CLI` 样本是分区式文本文件。官方 DSSAT《User's Guide Vol. 3》Appendix B 描述五个主要区段：站点/气候描述、SIMMETEO 月均值、WGEN 月分布参数、导入范围检查值和 flagged-data 计数。其定义包含：

- 气候/位置：`LAT`、`LONG`、`ELEV`、`TAV`、`AMP`、`SRAY`、`TMXY`、`TMNY`、`RAIY`、`START`、`DURN`、`ANGA`、`ANGB`、`REFHT`、`WNDHT`、`GSST`、`GSDU`。样本另出现 `INSI`、`SOURCE`。
- 月均值：`MTH`、`SAMN`、`XAMN`、`NAMN`、`RTOT`、`RNUM`、`SHMN`、`AMTH`、`BMTH`。
- WGEN：`MTH`、`SDMN`、`SDSD`、`SWMN`、`SWSD`、`XDMN`、`XDSD`、`XWMN`、`XWSD`、`NAMN`、`NASD`、`ALPHA`、`RTOT`、`PDW`、`RNUM`。
- QC/计数：`MIN/MAX/RATE` 对应逐天气变量的范围检查；flagged-data 区记录起止日期和 `TOTAL/VALID/MISSING/ERROR/ABOVE/BELOW/RATE` 计数。

含义、单位、尺度、来源及 requiredness 逐字段表见 [cli_field_structure.json](../results/yc_wgen_cli_pilot/003_06_03/cli_field_structure.json)。这些定义来自旧版 DSSAT v3 官方指南并由当前项目样本核对；不是 active DSSAT 4.8.0.024/4.8.5 的字段兼容性测试。对于 WGEN，旧指南明确要求气候文件中有 WGEN 参数；active 4.8.x 对最小字段集合的严格验证仍未知。不能手工拼出一个“看起来像”的文件。

## 4. YC 站点元数据与命名

只检查训练期 `CNYC0401.WTH` 至 `CNYC1301.WTH` 的头部字段。十个年度头部一致：`INSI=CNYC`、纬度 `36.830`、经度 `116.570`、海拔 `22`。WeatherMan 官方格式说明中坐标单位为度（北纬/东经）、海拔单位为米。`CNYC0801.MZX` 的 FileX `WSTA` 示例为 `CNYC0801`；对应历史 WTH 命名形如 `CNYC0801.WTH`，内部站点 ID 是 `CNYC`。

官方旧版 WeatherMan 新站点信息表还列出 Angstrom A/B、`REFHT/WNDHT`、TAV/AMP 和 growing-season start/duration，并说明站点信息填完前 export 不可用。十个训练期 WTH 头部虽重复含 `TAV=14.0` 与 `AMP=28.9`，但这些气候统计的参考期未知，因此不能直接作为 train-only 参数；`REFHT/WNDHT` 均为 `-99.0`，Angstrom A/B 与生长季字段未出现在头部。目标版本是否仍要求全部字段需看其 Help，缺失项不能用其他站点或猜值填补。

| 标识 | 结论 |
|---|---|
| 项目站点标签 | `YC`，由 YC 输入目录/实验上下文确认 |
| DSSAT 气象站 ID | `CNYC`，训练期 WTH 头部一致 |
| 经纬度/海拔 | 36.830°N，116.570°E，22 m；十年训练 WTH 头部一致 |
| 其他 WeatherMan 站点字段 | Angstrom A/B、growing-season start/duration 缺失；仪器高度为 `-99.0`；TAV/AMP 参考期未知 |
| `CN` 的语义 | 仓库没有正式定义，不擅自展开 |
| `YC` 作为 `CNYC` 子串的语义 | 仓库没有正式定义，不擅自展开 |
| 目标 CLI | `CNYC.CLI` |
| FileX 的年度 WSTA | 示例 `CNYC0801`，不是整个 `.CLI` basename |

官方公开 DSSAT 4.8.5 `SECLI.for` 对内部天气模式 `W`（WGEN）/`S`（SIMMETEO）把 `FILEW` 第 5–12 位替换为 `.CLI`，保留前四字符。因此对 `WSTA=CNYC0801`，静态源码推导的文件名是 `CNYC.CLI`。这确认了源代码层面的 basename 规则；并未运行当前 DSSAT，也未验证 active 安装的路径 fallback、文件内容 ID 检查或跨操作系统读取。

## 5. WeatherMan 的作用与输入格式

DSSAT 官方 Tools 页面将 WeatherMan 描述为日天气导入、分析、导出工具，支持导入原始数据/创建站点、编辑站点信息和生成天气。官方《User's Guide Vol. 3》中的 WeatherMan 参考章节描述菜单流程：选择 4 字符站码，导入一个或多个 daily weather raw files，建立导入格式（表头行、日期、变量、单位和列映射），写入站点 archive `.WTD`，选定起止日期后计算 WGEN 参数，并将 climate 参数保存到 `.CLI`。该指南是 DSSAT v3 资料，不能证明当前 4.8.x WeatherMan 版本的所有界面或兼容性。

针对冻结 CSV：

| 输入合同 | 当前结论 |
|---|---|
| 文件结构 | 逗号分隔 15 列；`DATE` 为 ISO 日期字段，另有 `YEAR/DOY` 与 source/QC 列 |
| 拟用于参数估计的日变量 | `SRAD,TMAX,TMIN,RAIN` |
| 目标变量单位 | WeatherMan 官方默认：SRAD MJ/m²、TMAX/TMIN °C、RAIN mm；CSV 表头没有 units row，导入前需按候选生成链核对并记录 |
| 时间尺度 | 日尺度，2004–2013 |
| WeatherMan 自定义列格式 | 旧官方指南描述可建立用户格式并配置日期、列和单位 |
| 逗号分隔符 / ISO 日期能否由目标版本直接读取 | 未确认 |
| 缺失值标记 | 冻结候选已完成 QC；目标版本对缺失标记的解析合同没有核实，不设置猜测 sentinel，也不启用自动填补 |
| 是否需要转换 | 未决；直接 CSV import 未验证。本轮不创建转换文件 |

若当前 WeatherMan 能用已验证的 custom format 直接读取该 CSV，应确保仅映射 `DATE` 与四个天气变量，其他 provenance 列不被解释为气象量。若不能，则只允许从冻结 CSV 进行新文件、逐值不变的确定性格式转换，并记录输入/输出 hash、3653 行、日期范围、列映射和单位转换。不得改写 frozen CSV 或用原始 `CNYC*.WTH` 代替冻结候选作拟合。

## 6. WeatherMan、版本与可访问性

| 问题 | 结论 |
|---|---|
| 官方工具角色 | WeatherMan 是 DSSAT 官方天气处理工具；是优先候选生成路线 |
| 导入 daily / 计算 WGEN / 保存 `.CLI` | 官方旧版指南说明了通用菜单流程和 `.CLI` 输出，但未由当前 4.8.x 本机界面复验 |
| GUI / batch / command line | 旧指南展示菜单交互；本仓库没有 WeatherMan CLI 估计器。当前版本是否支持 batch/CLI 未确认 |
| 本机 WeatherMan 安装/版本 | 未知；依 `AGENTS.md` 未探测项目外安装目录 |
| Windows 生成的文本 `.CLI` 能否直接供当前 Linux runtime 使用 | 文件样本是文本且 DSSAT 源码可构建于 Linux/Windows；但编码、换行、路径与实际读取均未做跨平台 smoke，不能标为已验证 |
| DSSAT 4.8.0 与 4.8.5 `.CLI` 差异 | 未确认；缺少一对同源版本输出和当前 runtime 检验 |
| 仓库内其他官方 CLI estimator | 未发现 |

WeatherMan 官方功能描述见 [DSSAT Tools](https://dssat.net/tools/)；导入与气候文件字段见 [DSSAT User's Guide Vol. 3](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf) 与 [Vol. 1](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol1.pdf)。命名依据为 DSSAT public source [`SECLI.for`](https://github.com/DSSAT/dssat-csm-os/blob/develop/InputModule/SECLI.for)，天气 mode `W` 在 [`SIMULATION.CDE`](https://github.com/DSSAT/dssat-csm-os/blob/develop/Data/SIMULATION.CDE) 中定义为内部 WGEN。指南版本与运行时版本不同，故只作流程与字段定义参考。

## 7. Gym-DSSAT 如何消费 `.CLI`

仓库参考 wrapper [`references/dssat_pdi.py`](../references/dssat_pdi.py) 中：调用方通过 `auxiliary_file_paths` 传入文件；wrapper 把它们按 basename 复制到新建临时运行目录；然后以该目录作为当前工作目录执行 DSSAT。YC/maize 没有自动附加 `.CLI` 的逻辑；例外是 wrapper 针对 cotton 的内置默认 CLI。wrapper 把 `random_weather=True` 转成 FileX weather mode `W`，并把 `rseed1` 写入 PDI YAML。

当前 YC safe-render 配置在 [`src/ppo_safe_rendering.py`](../src/ppo_safe_rendering.py) 明确使用 `random_weather=False`，只传 cultivar、历史 `.WTH` 与 soil，没有 `.CLI`。本轮没有启用该参数，也没有改训练 pipeline。静态集成路径可概括为：

```text
future isolated caller
  -> auxiliary_file_paths includes CNYC.CLI
  -> wrapper copies basename to temp run directory
  -> DSSAT runs with temp directory as cwd
  -> FileX WSTA=CNYC0801 + weather mode W
  -> public SECLI source derives CNYC.CLI from the four-character prefix
```

最后一步是公开 4.8.5 源码的静态规则，不是本机运行测试，也不证明 4.8.0.024 当前环境的实际 lookup。

## 8. 路线评估与阻塞项

| 路线 | 状态 | 理由 |
|---|---|---|
| A. WeatherMan | `BLOCKED_BY_TOOL_ACCESS` | 官方通用步骤有依据，但本机工具路径/版本不可检查，CSV import profile 未核实 |
| B. 其他 DSSAT official utility | `NOT_SUPPORTED_IN_REPOSITORY` | 项目内未发现明确的 `.CLI` 参数估计器；外部可用性未知 |
| C. 自行实现参数估计 | `NOT_IMPLEMENTED` | 前两项未明确无法使用，不得跳过官方路线；还需正式算法/方程依据 |

主要阻塞：

1. 当前 WeatherMan 的 About/version 与可用界面未知。
2. 冻结 CSV 的分隔符、日期编码和列/单位映射没有目标版本导入模板；目前不能决定直接导入还是先做转换。
3. 旧指南和跨站样本不足以确立 active DSSAT 4.8.x 的完整必需字段/可选字段。
4. 当前 active runtime 版本未知；历史保存运行的 DSSAT `4.8.0.024` 已确认，但不代表现时 runtime。
5. 当前 WGEN 文件查找虽有公开源码的静态 basename 规则，尚未做本机 lookup smoke。

## 9. 最小可执行下一步

1. 在不改系统/容器配置的前提下，由用户把 WeatherMan About/version 和该版本 Import/Export format/help 的文本或截图放入项目内 `results/yc_wgen_cli_pilot/003_06_03/user_provided_tool_info/`。
2. 按实际版本确认逗号分隔、ISO `DATE`、表头跳过、单位、列映射和缺失值处理。禁止默认自动填补。
3. 若该版本不能直接读 frozen CSV，再单独批准/实现从 frozen CSV 到它所支持 daily import format 的确定性转换；写到新目录，不覆盖原候选，输出行数/日期/hash/变量映射核对。
4. 在已确认的 WeatherMan 中建立新站点 `CNYC`，位置取一致的训练期 WTH header：36.830°N、116.570°E、22 m；按实际版本 help 确认其余字段，并从无验证泄漏的正式来源取得，不能照搬未知参考期 TAV/AMP、`-99` 高度或其他站点值。
5. 仅导入 frozen candidate 对应的 2004–2013 数据，明确选择 WGEN 参数并将起止期限定 2004/001–2013/365。保存为新目录 `results/yc_wgen_cli_pilot/003_06_03/generated/CNYC.CLI`，同时记录工具版本、源 hash、输出 hash、窗口及告警。
6. 静态审核输出站码、头部、12 个月 WGEN 字段和 provenance；确认 active DSSAT 版本后，再单独设计小规模 WGEN seed pilot 与 smoke gate。本任务不越级执行。

步骤 1–5 受上面的工具/导入格式阻塞；这是待版本确认的操作草案，不声称本轮已经运行。

## 10. Readiness 决策

```text
can_generate_cnyc_cli_now = BLOCKED_BY_WEATHERMAN_ACCESS
additional_blockers = BLOCKED_BY_UNKNOWN_IMPORT_FORMAT; BLOCKED_BY_UNKNOWN_CLI_FORMAT; BLOCKED_BY_UNKNOWN_STATION_METADATA
```

YC 核心站点 ID 与静态位置 metadata 已确认；`CN`/`YC` 子串解释、多个 WeatherMan 新站点字段、当前软件版本和实际运行匹配均未确认。没有使用任何其他站点参数。

## 11. 本轮输出、Git 状态和实验记录

- 报告：`docs/yc_cli_generation_path.md`
- 中文汇报：`docs/yc_cli_generation_path.pptx`
- 证据：`results/yc_wgen_cli_pilot/003_06_03/`
- 详细执行与失败尝试：[`experiment_log.md`](../results/yc_wgen_cli_pilot/003_06_03/experiment_log.md)
- 机器可读状态：[`cli_generation_readiness.json`](../results/yc_wgen_cli_pilot/003_06_03/cli_generation_readiness.json)
- 任务开始时上述输出路径不存在/干净；本轮仅暂存并提交下列交付文件。`.build/` 渲染暂存目录因清理操作被安全策略拒绝而保留，未纳入提交；既有无关工作树改动未纳入或覆盖。无 GitHub push，备份等待用户明确批准。

本轮文件：本报告、PPT、`cli_format_inventory.txt`、`cli_field_structure.json`、`yc_station_metadata_audit.json`、`weatherman_generation_path.txt`、`gym_dssat_cli_integration.txt`、`cli_generation_readiness.json`、`experiment_log.md`、PPT 构建脚本及验证记录。
