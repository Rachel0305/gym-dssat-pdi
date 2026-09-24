# YC WGEN Seed Pilot 与 Runtime Smoke（003_06_05）

**日期：** 2026-09-25（Asia/Shanghai）

**结论：** `FAIL_CLI_PARSE`。当前 DSSAT runtime 已找到并读取 YC `CNYC.CLI`，但在首个 WGEN 月参数记录处报错，未生成可比较的天气序列。本任务按失败门槛停止，没有继续运行其他 seed。

## 1. 范围与保护

- 仅测试 YC/CNYC；使用一份隔离的 YC 2008 FileX 副本、`MZCER048.CUL`、YC `SOIL.SOL` 和 train-only `CNYC.CLI`。
- 只尝试 `weather_seed=101`（`seed_101_run_a`）。计划中的 `101b`、`102–105` 未运行，因为 101a 在 WGEN 初始化阶段失败。
- 未启动 PPO；未修改现有 wrapper、PPO、reward、动作/观测空间或其他站点。
- `CNYC.CLI` 生成候选 hash 前后均为 `5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`；冻结训练天气 hash 前后均为 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`。未使用 validation weather。

## 2. Runtime 集成与候选传递

隔离调用链：

```text
run_yc_wgen_seed_pilot.py
  -> Gym-DSSAT maize environment
  -> auxiliary_file_paths: MZCER048.CUL, CNYC.CLI, SOIL.SOL
  -> isolated CNYC0801_wgen_template.jinja2
  -> random_weather=True; rendered FileX WTHER=W; WSTA=CNYC0801
  -> DSSAT runtime; weather_seed=101 written as rseed1_=101 in PDI YAML
```

证据见 `results/yc_wgen_cli_pilot/003_06_05/runtime/seed_101_run_a/cli_copy_evidence.txt` 及 `runtime_snapshot/`。运行目录内的 `CNYC.CLI` 与冻结候选逐字节 hash 一致；DSSAT 的 `ERROR.OUT` 也明确指向 `CNYC.CLI`，因此 lookup 成功。PDI YAML 已配置 `rseed1_=101`；由于 DSSAT 在首次天气生成前中止，无法确认该 seed 是否已被运行时消费。

本轮使用项目容器 `nifty_taussig`，DSSAT 在 `WARNING.OUT` 标记为 `4.8.0.024`。运行串行、单季、最多 400 个日步；容器开始时约有 14 GiB 可用内存。实际仅进入初始化，没有产生作物季节评价结果。

## 3. 运行结果

首轮输出：

```text
STOP 99
Unknown ERROR. Error number: 5010
File: CNYC.CLI   Line: 27   Error key: WGENIN
```

第 27 行是 `*WGEN PARAMETERS` 区段的第一个（月 1）数据行。DSSAT 在 2008/DOY 153 的 WGEN 初始化阶段终止；目前证据定位到该行记录，尚不能断定是其中哪个单字段或固定宽度/字段格式触发错误。没有依据归因为版本号不一致。

`WARNING.OUT` 另记录 FileX 纬度、经度、海拔为缺失值的警告，以及 soil `STONES`、`ADCOEF` 缺值后采用默认值。它们与 `ERROR.OUT` 中指向 CLI 的 fatal `WGENIN 5010` 分开记录；本报告不把这些非致命 warning 单独作为候选 CLI 失败理由。

## 4. Seed、天气捕获与 QC

| 检查 | 结果 |
|---|---|
| `weather_seed` 与 PPO seed 区分 | `weather_seed=101`；PPO seed=`NOT_APPLICABLE`。无 agent stochasticity 与 weather stochasticity 共用同一个实验 seed。 |
| PDI seed 接口 | `rseed1_=101` 已写入 PDI 配置；DSSAT 消费/使用 seed 未到达可验证阶段。 |
| 天气捕获 | 尝试从 runtime daily state 与生成 WTH 捕获；错误发生在首次天气状态之前，未产生日天气行或 WTH。状态为 `WEATHER_SEQUENCE_NOT_DIRECTLY_OBSERVABLE`。 |
| 同 seed 重复性 | 未测试；101a 未生成天气，因此未运行 101b。 |
| 不同 seed 多样性 | 未测试；102–105 未运行。 |
| 物理 sanity check | 未运行；无生成天气。训练期 2004–2013 冻结天气只通过 hash 检查，未用于拟合或调参。 |

当前 `reproducibility_check.json`、`seed_diversity_check.json` 与 `weather_summary_by_seed.csv` 保留未完成状态；不得将它们解释为通过。天气候选和 CLI 均未修改。湿日阈值差异不是本轮阻塞原因，也未在本轮做敏感性分析。

## 5. 兼容性决策与下一步

```text
runtime_cli_integration: CLI copied and hash verified; WTHER=W; WGENIN opened CNYC.CLI
runtime_compatibility_status: FAIL_CLI_PARSE
wgen_seed_pilot_status: INCOMPLETE_OR_FAILED
```

依据任务的输入保护规则，**不修改或重建原 `CNYC.CLI`，也不继续运行 101b/102–105**。下一步需先获得用户对“只诊断并修复第一个 WGEN 月记录的格式/字段，并产出新的隔离 CLI 副本”的明确批准。获批后，仅针对实际 `WGENIN 5010` 做最小修复与单次 smoke；兼容性通过后，再执行 101a/101b 和 102–105，验证天气捕获、同 seed 重复性、不同 seed 差异及基础 QC。全部 gate 通过后再进入 `003_06_06_yc_wgen_weather_qc_and_dssat_smoke`；本轮不进入 PPO。

## 6. 交付物与验证

- Pilot：`scripts/run_yc_wgen_seed_pilot.py`
- 单元测试：`tests/test_yc_wgen_seed_pilot.py`，`12 passed`；`py_compile` 通过
- PowerPoint：7 页；包完整性与布局验证通过，0 项发现；字体引用检查 152 项均为 Arial。未执行原生 PowerPoint 渲染验证。SHA256：`4FBA2949F6C19F50DB08E50152D55BEFB709ADAF9EBEF58EF9199807B589071C`
- Prompt 副本：`prompts/003_06_05_run_wgen_seed_pilot.md`
- 机器记录：`results/yc_wgen_cli_pilot/003_06_05/pilot_summary.json`
- Runtime 错误与输入副本：`results/yc_wgen_cli_pilot/003_06_05/runtime/seed_101_run_a/`
- 单次 seed 没有天气输出；无 generated daily weather CSV。
- Git：本任务文件单独暂存并以 `test: run YC WGEN seed pilot` 提交；不 push。GitHub backup pending explicit user approval。
