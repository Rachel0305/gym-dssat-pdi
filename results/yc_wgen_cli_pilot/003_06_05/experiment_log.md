# 实验记录：YC WGEN seed pilot（003_06_05）

## 任务与输入

- 日期：2026-09-25（Asia/Shanghai）；站点：YC/CNYC；运行模式：`random_weather=True`、FileX `WTHER=W`。
- 候选 CLI：`results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI`。
- 预期/运行前/运行后 SHA256：`5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0`，一致。
- Frozen train weather：`results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv`；前后 SHA256 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`，一致。没有读取 validation weather。
- YC 输入仅使用 `CNYC0801.MZX`、YC `MZCER048.CUL`、YC `SOIL.SOL` 与候选 CLI。isolated FileX 仅保留 2008 treatment 1；源输入未改。

## 调用与 seed

- Runtime：容器 `nifty_taussig`；DSSAT 自报版本 `4.8.0.024`。
- 调用链：pilot script -> Gym-DSSAT -> `auxiliary_file_paths` -> CLI basename `CNYC.CLI` 复制到运行目录 -> `WTHER=W`、`WSTA=CNYC0801` -> DSSAT WGEN。
- `weather_seed=101` 已写入生成的 `dssat-pdi.yml`，代码变量为 `rseed1_=101`。未使用 PPO；`ppo_seed=NOT_APPLICABLE`。运行在 WGEN 初始化失败，不能声称 DSSAT 已消费该 seed。
- 首轮运行约 32 秒后收到 PDI 等待超时。保留了这次运行的 `runtime_log.txt`、`CNYC.CLI`、`fileX.MZX`、`dssat-pdi.yml`、`ERROR.OUT`、`INFO.OUT`、`WARNING.OUT`、`RunList.OUT` 及 runtime 临时目录；仅终止了与此任务对应的挂起调用进程。

## 失败证据与决策

```text
STOP 99
Unknown ERROR. Error number: 5010
File: CNYC.CLI   Line: 27   Error key: WGENIN
```

- 第 27 行为第一个 WGEN 月参数记录（月 1）。错误发生在 2008/DOY 153、首次天气状态之前；未生成日天气记录或 runtime WTH。
- `WARNING.OUT` 另记录 FileX latitude/longitude/elevation 缺失及 `STONES`/`ADCOEF` 使用默认值。这些 warning 与 fatal `WGENIN 5010` 分开处理，没有把可选/非致命 warning 自动判为 CLI 失败。
- 兼容性：`FAIL_CLI_PARSE`。仅由首个错误即可定位到 WGENIN 对 CLI 首行的读取失败，尚不能进一步确定是哪一列或格式细节。
- 因 CLI 修改需后续用户明确批准，本轮不重建候选、不修 CLI、不继续其他 seed。

## Gate 状态

| Gate | 状态 |
|---|---|
| CLI hash 与 runtime 副本 | PASS |
| `WTHER=W`、`WSTA=CNYC0801` | PASS |
| WGEN CLI 解析与 runtime 启动 | FAIL_CLI_PARSE |
| `rseed1_=101` 配置 | PASS（仅配置证据；runtime 消费未验证） |
| 101 重跑 | NOT_RUN，首轮阻塞 |
| Seeds 102–105 | NOT_RUN，首轮阻塞 |
| 天气序列/相同 seed hash/不同 seed 差异 | NOT_TESTED，没有天气输出 |
| 物理 QC / 训练气候 sanity | NOT_RUN，没有天气输出 |
| PPO / 其他站点 / CLI candidate 修改 | NO |

单元测试：`python -m pytest tests/test_yc_wgen_seed_pilot.py -q`，12 passed；`python -m py_compile scripts/run_yc_wgen_seed_pilot.py tests/test_yc_wgen_seed_pilot.py` 通过。分析摘要已明确记录 `physical_sanity_status=NOT_RUN_NO_WEATHER_OUTPUT`，无天气输出时不将 QC 记作失败或通过。正式 WGEN pilot 总状态为 `INCOMPLETE_OR_FAILED`，不是 `WGEN_SEED_PILOT_PASS`。

PowerPoint：`docs/yc_wgen_seed_pilot.pptx`，7 页；包完整性、布局检查通过且无发现，152 个字体引用均为 Arial；未验证原生 PowerPoint 渲染。SHA256：`4FBA2949F6C19F50DB08E50152D55BEFB709ADAF9EBEF58EF9199807B589071C`。

## 下一步

等待用户明确批准后，针对 `CNYC.CLI` 第 27 行的 WGENIN 5010 做最小定点诊断/修复，使用新隔离 CLI 副本；先验证一个 runtime smoke，再继续 seed reproducibility/diversity。原 CLI 和冻结训练天气保持冻结。无 GitHub push；备份等待明确批准。
