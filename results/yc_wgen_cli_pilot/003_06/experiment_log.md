# 实验记录：YC train-only `.CLI` 与 WGEN pilot

- 执行日期：2026-09-24
- 分支：`codex/sya-forecast-freeze-2026-08-16`
- 开始 HEAD：`06a4b69fda7ec4c6542d5a258bd1c55631021a1e`
- 任务范围：YC/YCA；冻结既有 candidate；核对仓库内工具链和 0.0.5 调用合同。
- Final status：`BLOCKED_CLI_GENERATION`
- 汇报稿：4 页，PPTX 结构与布局检查通过；SHA256 `65d3744c04dcf14652eaf5385c4bf7f035eb859fc4495d085517fd962128b2d4`。未在原生 PowerPoint 中验证渲染。

## 已完成

1. 重新计算 candidate SHA256，并与上游 004 `upstream_gate.json` 冻结值核对，一致。
2. 复核 CSV 为 3653 行，日期连续覆盖 2004-01-01 至 2013-12-31；`RAIN/TMAX/TMIN/SRAD` 无空值；负雨量、负辐射、`TMAX < TMIN` 和重复日期均为 0。
3. 查阅 003_01 WGEN 审计、当前 wrapper 源码和 weather reset 诊断，确认此前没有合格的 train-only `.CLI`。
4. 记录既有旧 CLI 的排除理由：年份窗口包含 2014，估计年份 provenance 不完整，关联环境为 `random_weather=false`。

## 阻断与停止决定

项目内没有已验证的 WeatherMan 参数估计可执行文件或脚本。受项目 `AGENTS.md` 的目录边界约束，本轮不检查项目外安装位置，因此不能声称当前 WeatherMan 只能 GUI，亦不能确认其版本、路径或命令行能力。没有伪造 CLI，也没有套用旧 CLI。

CLI 未生成后停止后续昂贵步骤：WGEN calls=0，DSSAT smoke=0，PPO runs=0。天气 seeds 101–105 均未尝试；seed 101 复现和 101/102 差异均为 NOT RUN。`ppo_seed=NOT_USED`。

## 构建工具记录

- 第一次 PPT 脚本运行失败：`finalizePresentation` 重复声明。删除重复 import 后重跑。
- 第二次 finalization 未通过：缺少 `RUNTIME_NODE_MODULES` 环境变量，first-party Artifact Tool import 因而失败。设置工作区依赖路径后重跑，最终结构检查通过、布局警告数为 0。
- Candidate 核对无失败：SHA 与上一轮冻结值一致，日期序列连续，四变量完整，基础物理 QC 失败数为 0。

## 修改与保留

- candidate 天气文件未修改。
- 未改 PPO、reward、action、observation、生产 `WTH`、FileX、SOL、CUL，亦未触碰其他站点。
- 既有 `results/yc_wgen_cli_pilot/`（003_01）未覆盖，本轮文件放在独立的 `003_06/` 子目录。
- 开始前已有的用户改动与未跟踪研究产物全部保留。
