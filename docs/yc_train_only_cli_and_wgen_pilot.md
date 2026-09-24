# YC train-only `.CLI` 与 WGEN pilot / DSSAT smoke 记录

**最终状态：** `BLOCKED_CLI_GENERATION`

## 1. 结论

2004–2013 天气候选通过冻结检查，当前 SHA256 与前一轮 004 上游门槛记录一致。本轮没有生成可信的 train-only `CNYC.CLI`：项目内没有已验证的 WeatherMan 参数估计器或命令行流程；依照项目 `AGENTS.md`，本轮不检查项目目录以外的安装位置，因此也不能确认当前机器是否有可用的 WeatherMan GUI。旧 `CNYC.CLI` 的统计窗口延伸到 2014 年，且缺少仅以 2004–2013 估计的 provenance，故明确排除。

由于 CLI Gate 未通过，本轮没有运行 WGEN、seed reproducibility、pilot weather QC、DSSAT smoke 或 PPO。不得把这些未运行项表述为通过。

## 2. Candidate 冻结

| 项目 | 结果 |
|---|---|
| 站点 | YC/YCA |
| 文件 | `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` |
| SHA256 | `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34` |
| 大小 | 1,046,053 bytes |
| 日期 / 日数 | 2004-01-01 至 2013-12-31，3653 天 |
| 变量 | `RAIN/TMAX/TMIN/SRAD` 全部完整 |
| 物理 QC | 负降水 0、负辐射 0、`TMAX < TMIN` 0、重复日期 0 |
| 与既有冻结记录 | 与 `results/yc_ppo_weather_augmentation/upstream_gate.json` 中 candidate SHA256 完全一致 |
| 验证期排除 | 2014–2023；candidate 未修改 |

上游 QC 文件报告 3653 行、3653 个唯一日期、0 项物理失败；本轮对当前 CSV 再次检查了行数、日期范围、四个变量空值和上述物理条件。详见 `results/yc_wgen_cli_pilot/003_06/weather_candidate_freeze.json`。

## 3. DSSAT / WeatherMan / WGEN 工具链

- 仓库内记录的 `gym_dssat_pdi` 版本为 `0.0.5`，来源是既有 `results/yc_weather_audit/yc_weather_reset_diagnostic.json`。该记录给出历史容器内包路径 `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/`，本轮未在项目外重新探测运行环境。
- 仓库内 `references/dssat_pdi.py` 显示 `random_weather=True` 会请求 DSSAT 的 `W` 天气模式，并从 Gym 环境 RNG 抽取 `rseed1`。这只是源代码调用合同，不证明本机 WGEN 实际读取了某个 `.CLI`。
- 当前已安装 DSSAT 的版本、可执行文件路径、WeatherMan 版本/路径、WeatherMan CLI 或 GUI 能力，本轮均未能在项目范围内验证。历史目录名 `DSSAT480` 仅说明旧运行曾标记为 DSSAT 4.8，不作为当前版本证据。
- 当前 YC FileX 的 `WSTA` 到 `CNYC.CLI` 查找映射、`.CLI` 搜索路径、实际 DSSAT `RSEED` 及逐日 WGEN 天气导出路径都没有 runtime 证据。

完整盘点见 `results/yc_wgen_cli_pilot/003_06/toolchain_inventory.json` 和 `wgen_runtime_contract.json`。

## 4. `.CLI` provenance 与 Gate

没有生成 `results/yc_wgen_cli_pilot/003_06/cli_candidate/CNYC.CLI`，因此没有 CLI SHA256、参数估计器版本或可通过结构 QC 的文件；参数估计年份记为“未执行”，不能声称使用了 2004–2013。

项目内唯一已记录的历史候选位于旧 2000–2023 null-run 目录，SHA256 为 `58dbe11fdb3af9d34cc25644d8fb965627403778eb92da31a8f91619bd100d51`。文件记录窗口为 2008-01-01 至 2014-12-31；与之关联的 PDI 配置为 `random_weather=false`。它不满足无 validation leakage 的参数 provenance，不能改名或复制后用于本 pilot。

`.CLI` 状态为 `BLOCKED_CLI_GENERATION`；CLI 结构 QC 标记为未运行。未生成 GUI 手册，因为本轮无法证实当前工具链“只能通过 GUI 可靠运行”。

## 5. RNG 与 pilot 结果

本轮 `ppo_seed=NOT_USED`。计划天气 seeds 为 101–105，但没有任何 seed 被尝试。当前已记录的 0.0.5 wrapper 使用 Gym 环境 RNG 派生 `rseed1`，尚未实现或验证独立的 `weather_generation_seed` adapter；历史 weather reset 诊断不能代替本轮真实 WGEN 序列哈希。

| Gate | 本轮状态 |
|---|---|
| Candidate 冻结 | PASS |
| train-only `.CLI` 生成与 QC | BLOCKED / NOT RUN |
| seed 101-A 与 101-B 复现 | NOT RUN |
| seed 101 与 102 天气差异 | NOT RUN |
| WGEN pilot weather 数量 | 0 |
| pilot weather QC | NOT RUN |
| YC DSSAT 单季 smoke | 0 次，NOT RUN |
| PPO | 0 次，未启动 |

对应机器记录位于本轮目录的 `seed_reproducibility.json`、`weather_manifest.csv`、`weather_qc_by_seed.csv`、`weather_qc_summary.json` 和 `dssat_smoke_results.csv`。CSV 仅保留字段头，表示没有结果，不代表通过。

## 6. 最小后续步骤

先提供可在项目约束内使用的官方 WeatherMan 参数估计路径，或将工具版本、运行方式及合规的 candidate 导入/转换证据带入项目。估计输入必须唯一来自已冻结 candidate，严格限于 2004–2013；生成后把 `CNYC.CLI` 和完整 provenance 放到 `results/yc_wgen_cli_pilot/003_06/cli_candidate/`。随后先验证 `.CLI` 结构和 FileX `WSTA` 映射，再独立实现/验证 weather seed 与 PPO seed 分离、101 重复序列及 101/102 差异。只有这些 Gate 通过后才生成 3–5 套 pilot，并对至少 3 个 seed 做单季 DSSAT smoke。

本轮未训练 PPO；本轮结果不授权 `004_yc_ppo_weather_augmentation_experiment` 启动。

## 7. Git 与产物

- 开始 HEAD：`06a4b69fda7ec4c6542d5a258bd1c55631021a1e`；分支：`codex/sya-forecast-freeze-2026-08-16`。
- 保留开始前已修改的 5 个 tracked 文件与既有未跟踪研究产物；只暂存本任务专属文件。
- 完成审核后建议提交：`chore: validate YC train-only WGEN pilot`。未 push。
- 中文汇报稿：`docs/yc_train_only_cli_and_wgen_pilot.pptx`。
- 汇报稿 SHA256：`65d3744c04dcf14652eaf5385c4bf7f035eb859fc4495d085517fd962128b2d4`；PPTX 结构与布局检查通过，未在原生 PowerPoint 中验证渲染。
