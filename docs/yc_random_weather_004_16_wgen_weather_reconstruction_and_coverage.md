# 004_16 YC WGEN 天气重建与气候覆盖审计

## 结论摘要

本任务按用户设定的 provenance gate 停止在天气物化之前。**没有生成 WGEN 天气，没有运行 PPO、加载 checkpoint、执行 evaluation 或 DSSAT crop simulation。** 已完成历史生成链核查、输入哈希核验和存档盘点；另从 004_05 已保存的 observed WTH 计算了 2014–2023 作物季天气描述指标。由于没有可核验的 1001–1100 逐日 WGEN realization，本报告不对训练天气域覆盖、尾部缺口或 held-out 分布作正式结论。

```text
weather_reconstruction_status = FAILED
observed_training_coverage = INSUFFICIENT
tail_gap_type = INSUFFICIENT
weather_pool_expansion_recommendation = INSUFFICIENT_EVIDENCE
```

此处 `FAILED` 表示**重建 gate 未通过，物化未启动**，不是声称某个 WGEN 生成任务运行失败，也不表示 004_05 的训练作物结果无效。

## 样式与证据来源

本任务是数据来源审计，没有制作分布图。若 gate 通过，预定沿用项目白底、浅网格、紧凑的气候 QC 风格；当前不绘制训练/留出/observed 对比图，避免把未恢复的历史天气画成真实覆盖证据。空图目录中的 README 记录了这一 gate 决定。

核查文件包括：

- `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv`
- `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI` 与 `cli_generation_metadata.json`
- `results/yc_wgen_cli_pilot/003_06_05_02/runtime/formal_seed_pilot_provenance.json`
- `results/yc_random_weather_ppo/004_05/config/experiment_design.json` 与训练 schedule
- 004_05 seed5 training episode summary、observed/held-out evaluation manifest 与 episodes
- `results/yc_wgen_cli_pilot/004_01/full_year_path_audit/path_decision.json` 与 `source_trace.md`
- `results/yc_wgen_cli_pilot/003_06_02/runtime_environment.txt` 与 `audit_summary.json`
- 004_13 `attempt_05/formal_manifest.json`

输入文件的实际 SHA256、预期 SHA256 和检查结果见 `results/yc_random_weather_ppo/004_16_yc_wgen_weather_reconstruction/provenance/input_hash_inventory.csv`。

## 生成链还原

| 项目 | 状态 | 审计发现 |
|---|---|---|
| fitting weather | VERIFIED | frozen 2004–2013 fitting CSV 的 SHA256 与 CLI generation metadata 一致。 |
| frozen CLI | VERIFIED | `CNYC.CLI` 的字节 SHA256 与正式 seed-pilot provenance 一致。 |
| WGEN 参数 | VERIFIED | 冻结 CLI 是由上述 fitting 输入生成的 WGEN 参数文件；其存在不等于全年逐日导出器。 |
| 历史 DSSAT 版本 | INFERRED | 旧正式记录标注 DSSAT 4.8.0.024，官方 v4.8.0.24 源码被用于机制追踪；当前 runtime/binary 与二进制哈希未在本任务的项目归档中确认。 |
| WeatherMan / 独立 WGEN 工具版本 | UNKNOWN | 既有全年度路径审计将安装、API/CLI、CLI 导入兼容性均记为未知，没有通过的 full-year smoke。 |
| seed 调度 | VERIFIED | 004_05 schedule 的 100,000 行覆盖 1001–1080，每 seed 1250 次；seed5 episode 日志记录 scheduled seed 与发往 PDI 的 `RSEED1` 相同。 |
| seed 语义与可重放 RNG 状态 | UNKNOWN | initial `RSEED1` 可追踪，但没有完整内部 RNG state、版本锁定和脱离 crop simulation 的确定性逐日导出证据。 |
| 作物年份上下文 | VERIFIED | 训练 seed 与 `historical_year_context` (2004–2013) 配对；held-out context 固定为 2008。此上下文不可简化成仅有 seed 的无上下文映射。 |
| WTH 模板/格式化与 runtime hash 对象 | UNKNOWN | 存在少量 WTH 快照和逐 episode `runtime_weather_sha256`，但项目源码没有定义该 hash 究竟覆盖 WTH 原始字节、规范化日序列还是别的运行时载荷。 |
| 不运行 crop simulation 的物化路径 | UNKNOWN / 未验证 | 已审计的 DSSAT WGEN 通过每日 crop simulation loop 调用；已知路径会违反本任务禁止 crop simulation 的边界。没有确认可用的 standalone WeatherMan/WGEN runtime。 |

所以，**不能把相同 seed 编号加上相同 CLI 直接等同于恢复了 004_05 的原始逐日天气**。本次不调用 WeatherMan、不运行 DSSAT，也不另写近似 WGEN 实现。

## Hash 与归档核验

004_05 seed5 训练记录保存了 80 个 WGEN seed 的运行记录；观察到 170 个不同 runtime weather hash，说明同一个 seed 的天气哈希还与模拟上下文/运行记录相关，不能假设 seed→唯一全年天气文件。held-out 记录包含 1081–1100 的运行时哈希。文件级候选快照的 raw SHA 与 episode hash 未形成可证明的映射；规范化 daily-series hash 已对现存 WTH 快照计算并单独归档，但由于历史 hash 语义未知，不能把这两种 hash 直接作等价比较。

逐条记录见：

- `hash_verification.csv`：每条历史 runtime hash，状态为 `NOT_TESTED_NO_REGENERATION`，明确没有重建文件。
- `provenance/input_hash_inventory.csv`：fitting CSV、CLI、schedule、manifests 等来源 hash。
- `observed_wth_inventory.csv`：已有 observed WTH 的 raw 与 canonical daily-series SHA256。
- `weather_seed_file_manifest.csv`：1001–1100 seed 与上下文，生成文件字段为空、状态 `BLOCKED_PROVENANCE_GATE`。

## Observed 描述统计

现有 2014–2023 observed WTH 可以单独作描述性统计，不需要重跑 DSSAT。作物季起始日从对应 004_05 FileX 的 `PDATE` 读取，作物季长度取既有 evaluation episode 的 `episode_days`；DAP 阶段定义为 0–30、31–60、61–90、>90。无雨段定义为连续 `RAIN <= 0 mm`；复合 hot-dry 日定义为 `TMAX > 32°C` 且 `RAIN < 1 mm`。这些值不是 WGEN coverage 结论。

- 逐年指标：`observed_weather_metrics.csv`
- DAP 阶段指标：`stage_weather_metrics.csv`（仅 observed cohort）
- 复合极端：`compound_extreme_metrics.csv`（仅 observed cohort）
- seed5 重点年份 2014/2019：`seed5_key_year_weather_profile.csv`

若 observed 文件缺失或日期上下文不完整，须以对应表的 status 为准；不得以别的年份补值。

## 未完成与恢复条件

以下正式 coverage 输出因 gate 未通过而保持空表，且未绘图：

- `weather_metrics_by_realization.csv`
- `observed_training_percentiles.csv`
- `heldout_vs_training_distribution.csv`
- training/held-out/observed 三组季节分布图、DAP-stage coverage 图、observed percentile heatmap、seed5 key-year WGEN 对照图

恢复任务至少需要：

1. 可核验的 WeatherMan 或 standalone WGEN 工具版本，以及使用冻结 `CNYC.CLI` 的天气-only导出入口；
2. 对 crop-year/calendar/start date、seed与 RNG state 语义的可重复定义；
3. `runtime_weather_sha256` 的哈希对象/序列化规范，或带 seed/context 映射的历史 raw WTH；
4. 一条明确不启动 DSSAT crop simulation 的获批物化方式。

本轮没有证据支持 KEEP_80、随机扩容、定向尾部补样或修改 WGEN 分布。下一步决策保持 `INSUFFICIENT_EVIDENCE`；也没有启动 80→160 PPO 训练。Git 未 commit、未 push。
