# 004_17 YC runtime WGEN capture, historical weather recovery and permanent weather archiving

## 目标

只追溯并物化 004_05 实际使用的 YC WGEN 天气，先以 weather seed 1001 和历史 schedule context 做单样本 smoke。仅当历史 runtime weather SHA256 精确匹配或达到用户定义的可核验等级时，才串行恢复 1001–1100 并开展气候覆盖分析。建立今后随机天气实验的天气归档 helper 与 contract。

## 边界

- 不训练 PPO、不加载 checkpoint、不评估模型、不修改 canonical 配置、wrapper、reward、mask、observation、action space 或旧实验。
- 不将本任务触发的 DSSAT crop outcome 作为科学结果。只保留为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，且不输出产量/收益指标。
- 不覆盖旧文件；所有 runtime、天气、日志和分析文件写入 `results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/`。
- 先完成 source/provenance chain；随后只跑 seed1001 smoke。历史校验失败或不可验证时停止批量与正式 coverage。
- 固定容器 `/workspace`，宿主项目根 `C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi`；单环境、串行运行并监控 RSS。
- held-out seeds 1081–1100 保持 held-out。
- 不 commit、不 push。

## 历史 chain 与 hash

读取 004_05 schedule、run manifests、training logs、004_03 environment factory、DssatPdi runtime source snapshot、frozen `CNYC.CLI` 和 fitting CSV。逐项记录 WTHER、RSEED1 注入时点、crop-year/planting context、runtime temp path、WTH 是否实际落盘、daily state 捕获方式、cleanup 与覆盖行为。

复用 004_05 的 `runtime_weather_sha256` 算法：通过 `daily_weather_from_states` 产生 `DATE,DOY,RAIN,SRAD,TMAX,TMIN`，按日期/DOY排序、数值 `.12g`、UTF-8/LF 规范化后 SHA256。不得将其误称为 WTH raw-byte hash。

## Smoke 与批量 gate

从正式 schedule 精确恢复 seed1001 对应的历史 context，记录 CLI/runtime/Input provenance。生成实际 runtime weather 后立即复制任何存在的 WTH，并保存规范化逐日 CSV、metadata、raw/canonical hashes、生成参数及临时文件证据。将重建的 canonical runtime hash 与历史日志比较；同时检查历史 runtime snapshot/step trace 是否有可逐日比较天气。

状态为 `MISMATCH` 或 `UNVERIFIABLE` 时停止批量恢复和正式 coverage。只有 smoke 通过后才串行恢复 training 1001–1080 和 held-out 1081–1100；逐个 seed 写入归档、manifest 与日志。不得用相同 seed 编号代替实际 context/provenance。

若 DSSAT runtime 不产生磁盘 WTH，只能存档从实际每日 runtime state 捕获的 canonical series，并将 `raw_wth_status=NOT_EMITTED_BY_RUNTIME` 明确记录；不得伪造 runtime raw WTH。

## Coverage 与报告

仅当重建 gate 与逐日天气数据足够时，统一计算整季、DAP 0–30/31–60/61–90/>90 阶段、复合热旱、observed 2014–2023 相对 training 80 的经验百分位、支持范围及 held-out 分布差异。额外描述 seed5 的 2014/2019 天气关联，不作未经证实的因果归因。输出中文报告、CSV/JSON/PNG，并给出 `weather_recovery_status`、`observed_training_coverage`、`weather_tail_gap`、`next_weather_action`。
