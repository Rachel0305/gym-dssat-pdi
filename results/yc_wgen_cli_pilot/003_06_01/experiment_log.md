# 实验记录：YC WeatherMan / train-only `CNYC.CLI` 路径专项

- 日期：2026-09-24
- 分支：`codex/sya-forecast-freeze-2026-08-16`
- 开始 HEAD：`e056b9aefb46c9cd53b06b12d2a6aa90c69c68ba`
- Final status：`BLOCKED_TOOL_PATH_ACCESS`
- 任务边界：仅 YC/YCA；本轮不运行 PPO。
- 汇报 PPT：3 页，结构检查通过、布局警告 0；SHA256 `38F5820887F99769DC962ADCDA89F23029C5C4624F3215C78DBF3485AC6CA9F7`。未在原生 PowerPoint 中验证渲染。

## Gate A：candidate 复冻

实际重算 SHA256 为 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`，与任务要求一致。CSV 为 3653 行，首日 2004-01-01，末日 2013-12-31。未修改 candidate，也未纳入 2014–2023。

## Gate B：访问边界与停止决定

项目 `AGENTS.md` 明确规定只能读取和修改项目目录内文件。任务本身要求遵守此限制，因此没有检查 `C:\DSSAT*`、Program Files、PATH、Windows shortcut/registry、`/opt`、`/usr/local`、`/usr/bin` 或项目外 Python/Gym runtime。

由此无法判断 WeatherMan 是否安装、当前 DSSAT/WeatherMan 版本、可执行文件路径或 CLI/GUI 能力。状态是“未获准探测”，不是“未安装”。按任务 Gate B 输出 `BLOCKED_TOOL_PATH_ACCESS` 并停止；未查询导入格式、未确认 YC 元数据、未转换数据、未生成 CLI、未运行 WGEN/DSSAT/PPO。

## 官方资料

- DSSAT 工具页介绍 WeatherMan 可导入、分析和导出日天气，并列有创建站点、编辑站点信息、保存/导出等功能。此网页信息不代表本机安装版本的行为。
- 官方下载入口为 DSSAT Download System。项目当前 DSSAT 运行版本未在本轮核实，因此不建议仅为获取 WeatherMan 而直接切换 DSSAT 版本。

## 开始时的 Git 状态

- 开始时 HEAD 如上，工作树已有 5 个被修改的 tracked 文件及其他未跟踪研究产物；本任务未修改或暂存它们。
- 本轮只新增 candidate freeze、路径访问审计、实验摘要、中文报告与汇报 PPT。
