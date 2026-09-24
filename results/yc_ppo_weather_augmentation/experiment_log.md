# 实验记录：YC PPO 天气增强受控实验（004）

- 日期：2026-09-24
- 分支：`codex/sya-forecast-freeze-2026-08-16`
- 开始 HEAD：`a5a23852511cc9145f3e055b50977b3b5ba35381`
- 最终状态：`BLOCKED_UPSTREAM_WGEN_NOT_READY`
- 实验范围：YC 单站；train weather 为 2004-2013，validation 为 2014-2023；independent test 为 `not_available_or_not_verified`。

## 前置 Gate

本任务要求 `003_06_yc_train_only_cli_and_wgen_pilot` 达到 `PASS_YC_WGEN_PILOT` 后才能开始 PPO。本仓库未找到该任务专属结果目录、最终报告或 Gate 文件。可核验的既有 YC WGEN pilot 记录为 `BLOCKED_WGEN_NOT_READY`，CLI provenance 为 `BLOCKED_CLI_GENERATION`。

003_05_01 的天气 candidate 已通过并生成 3653 天，但该任务明确未生成 `.CLI`，未运行 WeatherMan、WGEN、DSSAT 或 PPO，因此不能替代 003_06 Gate。

## 已核实阻塞

- 当前 YC lowIC 输入目录没有 train-only `CNYC.CLI`。
- 唯一记录的旧 `CNYC.CLI` 标注窗口为 2008-01-01 至 2014-12-31，包含 validation 年份，且缺少仅以 2004-2013 拟合的 provenance，故排除。
- 旧记录未发现项目内可核实的 WeatherMan 参数估计程序；当前 FileX WSTA 与 CLI 的映射仍须运行时确认。
- 旧 pilot 已记录 WGEN realization 0、DSSAT WGEN smoke 0、PPO run 0。

机器可读证据见 `upstream_gate.json`。旧 CLI 哈希、源文件哈希与 candidate 哈希均已登记。

## 执行决定

- 不生成 synthetic weather bank，不运行 source sampler smoke。
- 不冻结或改写 PPO config，不建立 seed manifest，不启动 Stage 0 或 full training。
- 不运行 2014-2023 validation；两 Arms、paired metrics、seed success rate 均为 NOT_RUN。
- 未选择或剔除 PPO seed。当前状态只是 upstream 阻塞，不代表天气增强实验效果好或差。
- 已有 221YCA 历史天气场景增强实验不属于 train-only WGEN 实验，不作为本任务结果或 upstream PASS 证据。
- 四页状态 PPT 已生成；包完整性、布局检查通过，逐页预览已查看。原生 PowerPoint 字体渲染未验证。

## 下一步

先完成 003_06：使用严格限制在 2004-2013 的 YC 天气 candidate 生成并审计 CNYC.CLI provenance，确认 WSTA 映射，运行独立天气 seed 的 WGEN 复现检查与 DSSAT smoke。只有 003_06 最终状态达到 `PASS_YC_WGEN_PILOT` 后，再恢复 004 的 synthetic bank、两阶段 PPO 和统一 validation 流程。
