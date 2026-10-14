# 004_23B YC 外部实测天气注入 smoke

## 结论

- `external_weather_injection_status = FAILED`
- `wth_copy_status = BYTE_IDENTICAL`
- `daily_weather_match_status = MISMATCH`
- `wgen_bypass_status = SUPPORTED`
- `next_step = FIX_INJECTION_PATH`
- 重复性：`NOT_TESTED`（按单次最小 smoke 设计，未做第二次 episode）
- 日期/DAP 映射：`NOT_ASSESSED`。

## 输入与 provenance

- 样本：YC observed weather，calendar year `2004`；使用既有 004_03 historical-weather run 的输入上下文，未按天气极端性选择。
- 源 WTH：`/workspace/results/yc_external_weather_augmentation_004_24B/runtime_io_candidate_001_inputs/attempt_003/CNYC0401.WTH`，SHA256 `BD5272E6E9F8B670EC5B3B6E4E99DDF1B0E775147B92757EFCC73E3CCCDA68D1`。
- 源 FileX：`/workspace/results/yc_random_weather_ppo/004_03/training_verified/historical/ppo_seed_0/rendered_inputs/YCA/2004/HISTORICAL_WEATHER_ppo_seed_0_2004/YCA_2004_HISTORICAL_WEATHER_ppo_seed_0_2004.jinja2`，SHA256 `775B81FE074115CE0D5A243583F6032CFD2FFB7A94FAA51FE4BA5D3742EC2E53`。
- FileX：WSTA `CNYC0401`，PDATE `04169` (`2004-06-17`)，SDATE `04165` (`2004-06-13`)，WTHER `M`。
- 源 WTH 与 004_03 renderer 既有副本 byte-identical：`False`。

## Runtime 核验

- Gym-DSSAT 参数：`random_weather=False`；runtime FileX 的 WTHER=`M`。
- runtime 实际 WTH：`/tmp/tmpc0so5cj3/CNYC0401.WTH`；启动后拷贝 SHA256 `BD5272E6E9F8B670EC5B3B6E4E99DDF1B0E775147B92757EFCC73E3CCCDA68D1`，episode 后 SHA256 `BD5272E6E9F8B670EC5B3B6E4E99DDF1B0E775147B92757EFCC73E3CCCDA68D1`。
- 原始 WTH 与 runtime copy：`BYTE_IDENTICAL`。
- runtime 日天气行数 `102`；按日期键 join `102` 行 / `102` 个唯一日期（join 不代表数值匹配）；未映射 `0` 行；输入外日期 `0`；重复 runtime 日期 `0`。
- 最大绝对差：SRAD `21.600000381469727`、TMAX `7.50000076293945`、TMIN `6.0`、RAIN `80.8`；严格容差 `1e-05`。
- 日期键优先使用 PDI 的 YRDOY；具体日期映射见 `runtime_weather_daily.csv`。
- runtime 日期序列分段数：`0`；详情见 `execution_record.json`。固定偏移诊断中 `source_date = runtime_YRDOY - 1 day` 匹配 `NA` / `NA` 行，仅作诊断，不自动接受。
- 生命周期诊断：未发现额外 runtime 生命周期问题。
- PDI YRDOY 同日直接比较状态为 `MISMATCH`；只有源码/运行时序能解释固定偏移后，才会另行闭合日期映射并重判。
- Gym-DSSAT/PDI 时序审计：`NOT_REVIEWED`。wrapper 收发时序与仍未知的 DSSAT/PDI 引擎内部日历时序分别记录于 `runtime_state_timing_audit.md/json`。
- WGEN bypass 判断依据：`random_weather=False`、WTHER=M、runtime copied WTH 哈希和逐日比较；判定 `SUPPORTED`。

## 实验边界

本次没有训练 PPO、运行 WGEN 或生成 synthetic weather。Gym-DSSAT 构造后直接从 `raw._state` 捕获首状态，未调用 `raw.reset()`，随后只运行一个固定 no-op episode。所有 crop、yield、stress、Summary.OUT 等均为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，不作农艺解释。runtime 临时 WTH 在 cleanup 前已复制至本任务结果目录。

## 产物

- 此报告针对 `/workspace/results/yc_external_weather_augmentation_004_24B/runtime_io_candidate_001/attempt_003` 中的 clean run；较早 attempt 的失败/重复轨迹保留在其独立目录，不并入本次统计。
- `source_input_manifest.json`
- `runtime_file_inventory.csv`
- `input_weather_daily.csv`
- `runtime_weather_daily.csv`
- `weather_loop_comparison.csv`
- `runtime_date_offset_diagnostic.csv`
- `weather_loop_comparison_minus1_day_diagnostic.csv`（诊断用途，不自动作为 match）
- `runtime_state_timing_audit.md/json`
- `execution_record.json`
- `decision.json`
