# 实验记录：YC WGENIN 5010 格式修复（003_06_05_01）

## 目的与约束

- 日期：2026-09-25（Asia/Shanghai）。仅诊断 YC/CNYC `WGENIN 5010`，只允许一个 `weather_seed=101` runtime smoke。
- 不运行额外 seed 或 PPO；不重算 WGEN 参数，不修改冻结天气、其他站点、FileX/土壤输入。
- 冻结源 CLI：`results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI`。

## 诊断与修复

- 原 runtime：DSSAT 4.8.0.024，`WGENIN 5010`，`CNYC.CLI` 第 27 行（月 1）。冻结 CLI SHA256 前后均为 `5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`。
- WGENIN 格式合同：`(I6,14(1X,F5.0))`，共 15 列，90 字符宽；MTH 为整数，另 14 个统计值为实数。v4.8.0.24 与 v4.8.5.0 公共源码 READ/FORMAT 相同。
- 原 formatter 的 `XDMN`、`XWMN`、`NAMN` 字段宽 7，其余统计字段宽 6；原月行 93 字符，造成后续列错位。
- 生成器现统一输出 `I6 + 14*(1X,F5.0)`。修复仅改变 12 条 WGEN 行的空格宽度，所有数值 token 原样保留。
- 隔离候选 SHA256：`65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`。runtime 专用 basename 副本 `candidate/runtime_input/CNYC.CLI` 与候选 hash 相同。

## 第一次包装器尝试

第一次把长文件名作为辅助文件传入，运行目录没得到预期 basename `CNYC.CLI`，DSSAT 在 `MAKEFW`、line 0 报找不到天气文件，没有到达 WGENIN。该尝试保留于 `runtime/attempt_01_basename_lookup_failure.json`，重新标记为 `NOT_TESTED`，不把“未出现 5010”误报为修复成功。之后使用 hash 相同的 runtime basename 副本执行唯一一次有效 WGEN smoke。

## 静态测试

- 12 个月 WGEN 数据行均 90 列，月序 1-12；无非有限值、格式溢出或字段错位。
- 修复前后 14 个统计值的 token 完全一致，未拟合或修改气候参数。
- `python -m pytest tests/test_build_dssat_cli.py tests/test_yc_wgen_seed_pilot.py -q`：23 passed。
- 四个本任务 Python 文件 `py_compile` 通过。

## 有效 smoke 结果

- 仅 `weather_seed=101`，`ppo_seed=NOT_APPLICABLE`；YC 单季、`random_weather=True`、FileX `WTHER=W`、`WSTA=CNYC0801`。
- DSSAT 4.8.0.024，runtime CLI basename `CNYC.CLI`，runtime hash 与候选一致。
- 运行 120 步，捕获 120 行天气状态，日期 2008-06-01 至 2008-09-28；物理 screening QC 为 `PASS`。
- `runtime_compatibility_status=PASS`，`runtime_smoke_status=PARSE_FIX_RUNTIME_PASS`，`WGENIN 5010` 已通过该 smoke 验证解决；没有继续其他 seed。
- frozen CLI 与 train-weather hashes 前后不变；没有 PPO、其他站点或 validation weather 运行/修改。
- WARNING.OUT 仍记录经纬度/海拔读取及变量传递 warning，本任务未扩大范围处理。

## 产物与版本控制

- 中文报告：`docs/yc_wgen_5010_parse_fix.md`
- 中文 PPT：`docs/yc_wgen_5010_parse_fix.pptx`
- 结构、逐列和 hash 证据：本目录下 `diagnostics/`、`candidate/`、`validation/`、`runtime/`。
- 本地 commit message：`fix: align YC CLI with WGENIN parser`；只暂存本任务记录与生成器/回归测试改动。SHA 见最终终端摘要；未 push。GitHub 备份等待明确批准。
