# YC WGENIN 5010 定点诊断与修复

## 任务范围

本轮只修复 YC/CNYC 隔离 CLI 副本的 WGEN 月记录格式，并使用 `weather_seed=101` 做一次 runtime smoke。没有改动冻结 CLI、训练天气、拟合公式、其他站点或 PPO；没有运行其他 seed。

## 原始失败

上一轮 DSSAT runtime 为 4.8.0.024，报告 `WGENIN 5010`，文件 `CNYC.CLI` 第 27 行，即第一个月度 WGEN 记录。冻结候选 `results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI` 的 SHA256 为 `5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`，本轮前后复核一致。

## 读取合同与逐列布局

DSSAT 公共源码在 v4.8.0.24 和 v4.8.5.0 的 `WGENIN` 均以 `(I6,14(1X,F5.0))` 读取月记录：[DSSAT v4.8.0.24 WGEN.for](https://github.com/DSSAT/dssat-csm-os/blob/v4.8.0.24/Weather/WGEN.for#L408-L414)、[DSSAT v4.8.5.0 WGEN.for](https://github.com/DSSAT/dssat-csm-os/blob/v4.8.5.0/Weather/WGEN.for#L412-L418)。当前 runtime 4.8.0.024 与 v4.8.0.24 标签对应的读取格式一致，且实际 runtime smoke 通过，因此本次检查范围内的跨版本合同兼容性记为 `SUPPORTED`。合同共 15 列：`MTH` 一个整数列组，加 14 个 REAL 统计值；记录总宽 90 列。

下表位置采用 1-based 列号。渲染值取修复后 1 月记录；CNSY 仅用于结构比对，没有复制其气候参数数值。

| 位置 | 字段 | 预期类型 | 宽度/格式 | 修复后值 | 修复后字段 | 兼容 |
|---|---|---|---|---:|---|---|
| 1-6 | MTH | INTEGER | 6 / I6 | 1 | `     1` | 是 |
| 8-12 | SDMN | REAL | 5 / 1X,F5.0 | 7.6 | `  7.6` | 是 |
| 14-18 | SDSD | REAL | 5 / 1X,F5.0 | 3.2 | `  3.2` | 是 |
| 20-24 | SWMN | REAL | 5 / 1X,F5.0 | 5.0 | `  5.0` | 是 |
| 26-30 | SWSD | REAL | 5 / 1X,F5.0 | 2.8 | `  2.8` | 是 |
| 32-36 | XDMN | REAL | 5 / 1X,F5.0 | 3.3 | `  3.3` | 是 |
| 38-42 | XDSD | REAL | 5 / 1X,F5.0 | 3.6 | `  3.6` | 是 |
| 44-48 | XWMN | REAL | 5 / 1X,F5.0 | 1.9 | `  1.9` | 是 |
| 50-54 | XWSD | REAL | 5 / 1X,F5.0 | 4.0 | `  4.0` | 是 |
| 56-60 | NAMN | REAL | 5 / 1X,F5.0 | -7.2 | ` -7.2` | 是 |
| 62-66 | NASD | REAL | 5 / 1X,F5.0 | 3.6 | `  3.6` | 是 |
| 68-72 | ALPHA | REAL | 5 / 1X,F5.0 | 0.811 | `0.811` | 是 |
| 74-78 | RTOT | REAL | 5 / 1X,F5.0 | 1.9 | `  1.9` | 是 |
| 80-84 | PDW | REAL | 5 / 1X,F5.0 | 0.023 | `0.023` | 是 |
| 86-90 | RNUM | REAL | 5 / 1X,F5.0 | 0.9 | `  0.9` | 是 |

原始第 27 行长 93 字符，头行长 90。逐列套用读取格式时，`XDMN` 字段已越过预期 5 字符值宽，随后 `XDSD` 的固定列片段出现非数值拼接；这是对列布局的静态诊断，不把它伪称为运行时 Fortran 的具体 IOSTAT 解释。仓库 `CNSY.CLI` 的完整 WGEN 月行长 90，header 与列次序相同。完整逐字符证据见 `results/yc_wgen_cli_pilot/003_06_05_01/diagnostics/original_line27_layout.txt` 和 `wgen_row_schema_comparison.txt`。

## 根因与最小修复

原生成器在 `scripts/build_dssat_cli.py` 将 `XDMN`、`XWMN`、`NAMN` 输出为宽 7 的数值字段，其余统计值为宽 6；WGENIN 要求每个统计值占一个空格分隔符加 F5，共 6 列。三个字段各多一列，使每行由 90 列变为 93 列并导致后续固定列错位。修复只统一 WGEN formatter 到 `I6 + 14*(1X,F5.0)`，没有重拟合天气、改变湿日阈值或调整数值。

隔离候选：`results/yc_wgen_cli_pilot/003_06_05_01/candidate/CNYC_parsefix_01.CLI`，SHA256 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`。只改动行 27-38 的空格/字段宽度；12 个月所有参数 token 与源文件相同，`parameter_values_changed=NO`。runtime 专用副本以 `CNYC.CLI` basename 放在 `candidate/runtime_input/`，其 hash 与隔离候选相同；冻结原件未覆盖。

## 静态检查与回归测试

`validation/static_parse_check.json` 记录：12 行均为 90 列、15 列全部存在、数值有限且可解析、没有 NaN/Inf/宽度溢出，源 token 完全保留，结果 `PASS`。新增 formatter/解析及源值保留测试，并运行：

```text
python -m pytest tests/test_build_dssat_cli.py tests/test_yc_wgen_seed_pilot.py -q
23 passed
```

`py_compile` 对生成器、相关测试、候选生成脚本和 runtime smoke 包装脚本均通过。

## Seed 101 runtime smoke

第一次包装器尝试因把长文件名 `CNYC_parsefix_01.CLI` 直接传给运行器，DSSAT 在 `MAKEFW` 阶段找不到所需 basename `CNYC.CLI`，没有进入 WGENIN。该次证据保留在 `runtime/attempt_01_basename_lookup_failure.json`，并明确标为 `NOT_TESTED`，不计为一次 WGEN 解析结论。随后将同 hash 的隔离副本放入本任务 `candidate/runtime_input/CNYC.CLI`，只执行一次有效 seed 101 smoke。

有效运行的配置为 YC 单季、`random_weather=True`、FileX `WTHER=W`、`WSTA=CNYC0801`，无 PPO。DSSAT 运行 120 步，runtime 读取的 CLI hash 与修复候选相同，捕获 120 行（2008-06-01 至 2008-09-28）天气状态；没有 `WGENIN 5010` 或其他 runtime error。状态为 `PARSE_FIX_RUNTIME_PASS`，天气字段 screening QC 为 `PASS`。本次 smoke 仅验证 CLI 可通过 WGEN 初始化并进入后续模拟，不构成多 seed 随机性或天气气候学验证。运行摘要、日志、runtime 快照和生成天气 CSV 均保存在本任务 `runtime/` 与 `generated_weather/`。

Warning 文件仍记录 FileX latitude/longitude/elevation 读取与字段传递 warning；它们没有阻止本次模拟。本轮按任务范围未修改 FileX 或土壤输入。

## 结论与下一步

根因已定位为 formatter 对 `XDMN`、`XWMN`、`NAMN` 多输出一列，格式修复已在隔离候选上通过静态检查、回归测试和一个 seed 101 runtime smoke。冻结原 CLI 和训练天气 hash 均前后不变。按任务 gate 停止，不运行 `101b` 或 `102–105`；推荐下一任务恢复完整 003_06_05 seed pilot。

## 产物与版本控制

- 详细逐列布局：`results/yc_wgen_cli_pilot/003_06_05_01/diagnostics/`
- 修复候选与 diff：`results/yc_wgen_cli_pilot/003_06_05_01/candidate/`
- Smoke 汇总：`results/yc_wgen_cli_pilot/003_06_05_01/runtime_smoke_summary.json`
- 中文幻灯片：`docs/yc_wgen_5010_parse_fix.pptx`
- Git status：工作区另有与本任务无关的既有变更，未清理、未回退；本任务仅暂存本任务产物以及生成器和回归测试改动。
- Git commit：本地提交信息为 `fix: align YC CLI with WGENIN parser`；提交 SHA 在最终终端摘要提供。未执行 push。GitHub 备份等待用户明确批准。
