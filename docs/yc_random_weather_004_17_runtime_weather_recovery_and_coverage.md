# YC 004_17 runtime WGEN 天气恢复与气候覆盖审计

## 范围与边界
本任务未训练 PPO、未加载 checkpoint、未运行策略评估。为捕获天气而执行的 DSSAT no-op episode 只作为 runtime 天气物化手段；其作物模拟输出属于 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，不进入任何性能统计。旧实验目录未修改。

## 生成链与历史 hash
完整逐层来源见 `results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/provenance/runtime_weather_generation_chain.md`。历史 SHA256 是 PDI daily-state 序列化哈希，不是 WTH raw-byte hash。历史 pilot 的 runtime snapshot 枚举了可见输出文件但没有 `.WTH`；004_05 rendered input WTH 是 observed 源文件副本，不能视为 WGEN 实现。

## Seed1001 smoke
- 状态：`EXACT_MATCH`；seed=1001、crop-year context=2013、schedule episode=26。
- 历史 hash：`13C121A6DFED924FAB7C38EC53DDCC6FB37A39B0B1D2578DDBE42D445D53FEF1`；生成 hash：`13C121A6DFED924FAB7C38EC53DDCC6FB37A39B0B1D2578DDBE42D445D53FEF1`。
- raw WTH：`NOT_EMITTED_OR_NOT_CHANGED_FROM_INPUT`。仅变化/新生成于 runtime 的 WTH 才会复制；未将输入 WTH 冒充输出。
- 初次新建进程、直接 bootstrap seed1001 的诊断尝试未复现缓存 runtime 上下文（309 captured states，hash 不匹配），已单独保留且不作为正式验证。正式 smoke 重放 2013 year-runtime 的 bootstrap seed1026 及前序 RSEED1=1026、1043 后，对 seed1001 逐日 canonical hash 精确匹配（103 captured rows）。

## 批量恢复
- context replay 记录数：100。最终逐 seed 对照见 `hash_verification_context_replay_reconciled.csv`；`hash_verification.csv` 保留运行尝试记录，不应与最终 reconciled ledger 混读。
- 已归档 training realization：80/80；held-out：20/20。
- 历史 hash reconciliation：training exact 78/80；held-out exact 20/20。held-out 依据 004_05 正式 evaluation episode summary 对照；带 RSEED1/历史 hash 的最终映射见 `weather_archive_manifest_verified.csv`。
- 独立 host-shell 完整性复核：权威 manifest 的 100/100 个 canonical CSV 均存在且文件 SHA256 与清单一致，详见 `provenance/final_archive_integrity.json` / `.csv`。全尝试归档保留 134 个 canonical 文件及 137 条 episode manifest 记录；其中只有 reconciled 100 行 manifest 用作目标 weather-seed 对照，其余诊断尝试保留但不用于覆盖分析。
- runtime 未产生新的/变化的 WTH 文件（归档 `.WTH` 数量为 0）；归档内容是实际 daily-state weather series，不冒称 raw WTH。未来 helper 的 manifest 已纳入 `RSEED1` 字段。
- 未通过逐日历史 hash 的记录：seed=1036 / crop-year=2009 / rows=100 / generated=AFCE25700AD1CEF9DC397A4770C17E23DE0F0FF5D186CE8E0B50A6A1D277DD3D / historical=5FE6CE48C76ACC3DF2DA32DEEF6050C91B1684DA6D69864ED083BC694E22B0B4。
- 未通过逐日历史 hash 的记录：seed=1003 / crop-year=2011 / rows=108 / generated=EC8FA734477ED789D8EDD11331C76563CD5F42D3BC52131B7850270556FC1BDB / historical=5593D99D25DE5DC0E33935F2239558DCF96BB710F07007F2D46C670005D2ABE9。
- 这两条 004_05 历史 episode log 分别记录 102/110 episode days，而当前 no-op capture 得到 100/108 daily rows；长度差异可能与 episode 终止/状态记录范围有关，但缺少历史逐日 weather snapshot，不能据此宣称只是末尾截断或数值相同。
- 若 smoke gate 未通过，不启动批量恢复或正式 coverage。批量 hash 不完整或不匹配时，不得把整体结果称为 exact historical reconstruction；未通过记录按未验证处理。

## Coverage 与限制
- `observed_training_coverage = INSUFFICIENT`
- `weather_tail_gap = INSUFFICIENT`
- `next_weather_action = INSUFFICIENT`
- Coverage状态：INSUFFICIENT。Observed weather 来自 004_05 seed5 的官方 evaluation daily-state trace，按 year/DAP 去重；没有用产量解释天气。
- 本轮正式 climate coverage 未执行：历史逐日 hash gate 未完整通过，且没有物化齐全的 80/20 exact historical series。
- 因此没有生成正式的 realization/stage/compound 指标表、observed percentile heatmap 或 climate distribution figures；不据这 98 条已核验序列作正式 coverage 外推。

## 最终状态
`weather_recovery_status = PARTIALLY_VERIFIED`
`observed_training_coverage = INSUFFICIENT`
`weather_tail_gap = INSUFFICIENT`
`next_weather_action = INSUFFICIENT`

本报告未把运行时 side-effect 的 crop outcome 当科学结果。
