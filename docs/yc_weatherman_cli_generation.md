# YC WeatherMan / train-only `CNYC.CLI` 生成路径专项

**最终状态：** `BLOCKED_TOOL_PATH_ACCESS`

## 结论

本轮完成了 candidate 复冻，但无法在现有项目安全边界下确认当前机器是否安装 WeatherMan。项目 `AGENTS.md` 明确禁止访问项目目录之外的文件；本专项也明确要求遵守该限制，遇到时输出 `BLOCKED_TOOL_PATH_ACCESS` 并停止。因此本报告不把“未知”写成“未安装”。

没有检查外部安装路径、PATH、Windows shortcut/registry 或项目外 runtime。也没有继续猜 WeatherMan 导入格式、填 YC 站点元数据、构造 import package 或生成 `.CLI`。未运行 WGEN 或 PPO。

## 1. Frozen candidate

| 项目 | 结果 |
|---|---|
| 文件 | `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` |
| SHA256 | `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34` |
| 日数 | 3653 |
| 日期 | 2004-01-01 至 2013-12-31 |
| 与任务预期 | 一致 |
| Candidate 修改 / 2014–2023 混入 | 均无 |

机器记录：`results/yc_wgen_cli_pilot/003_06_01/input_freeze.json`。

## 2. WeatherMan / DSSAT 检查状态

| 检查项 | 状态 |
|---|---|
| WeatherMan 是否安装 | 未知，未探测 |
| WeatherMan 可执行路径 / 版本 | 未核实 |
| DSSAT 当前安装路径 / 版本 | 未核实 |
| CLI / GUI 能力 | 未核实 |
| 当前版本实际导入格式 | 未核实，Gate C 未启动 |
| CNYC 经纬度 / 海拔等元数据 | 未核实，Gate D 未启动 |
| import package / round-trip QC | 未创建 / 未运行 |
| `CNYC.CLI` | 未生成，无 hash 或 provenance |

上一轮项目记录中的 `gym_dssat_pdi 0.0.5` 和历史 `DSSAT480` 目录名不能证明当前本机安装版本。本轮没有据此推断兼容性。

## 3. 为什么停止

本轮任务列出的只读探测范围包括 `C:\DSSAT48\`、`C:\DSSAT*`、`C:\Program Files\DSSAT*`、PATH、Windows shortcut/metadata、`/opt/`、`/usr/local/`、`/usr/bin/` 和项目外 Python/Gym runtime。`AGENTS.md` 明确禁止读取项目目录外的文件，因此这些位置均未检查。机器记录见 `results/yc_wgen_cli_pilot/003_06_01/weatherman_inventory.json`。

DSSAT 官方工具页说明 WeatherMan 可导入、分析和导出日天气，涵盖创建 climate station、编辑 station 信息及保存/导出；这只是官方产品说明，不能替代对本机版本 help/About 的核验。[DSSAT Tools](https://dssat.net/tools/)

官方软件下载入口为 [DSSAT Download System](https://get.dssat.net/)。由于项目当前 DSSAT runtime 版本未知，不建议为了得到 WeatherMan 而未经兼容性确认直接切换 DSSAT 版本。

## 4. 最小人工核验清单

请在你自己的机器上只读查找，并把结果文本或截图复制到项目内，例如 `results/yc_wgen_cli_pilot/003_06_01/user_provided_tool_info/`：

1. WeatherMan executable 的完整路径和文件名；若找不到，请明确说明检查过的安装盘/开始菜单入口。
2. DSSAT 安装根目录与实际运行时版本；优先提供 DSSAT 输出文件版本头或 About 信息。
3. WeatherMan 的 About/version 信息截图或原始文本。
4. 若 Help/About 显示命令行参数或 batch 接口，请同时提供原始 help 输出；不要运行生成操作。

仅有路径名或文件名不足以认定版本。拿到这些项目内证据后，下一轮才继续核验当前版本接受的天气导入格式、YC 元数据和 candidate 转换流程。参数估计仍必须只用 frozen candidate 的 2004–2013 数据；不使用旧 2008–2014 CLI，不纳入 2014 年以后天气。

## 5. 当前 Gate 与下一步

- Candidate freeze：`PASS_INPUT_FREEZE`
- WeatherMan 路径发现：`BLOCKED_TOOL_PATH_ACCESS`
- 导入合同 / 站点 metadata / 转换 QC：未执行
- CLI / WGEN / DSSAT / PPO：未执行
- 中文汇报 PPT：`docs/yc_weatherman_cli_generation.pptx`；结构检查通过，布局警告 0；原生 PowerPoint 渲染未验证。

下一最小任务是把 WeatherMan executable 路径、DSSAT 安装根目录和 About/version 证据放到项目目录内，再按实际版本核验。所有新结果保存在 `results/yc_wgen_cli_pilot/003_06_01/`。本轮未 push。

完整实验记录：`results/yc_wgen_cli_pilot/003_06_01/experiment_log.md`。
