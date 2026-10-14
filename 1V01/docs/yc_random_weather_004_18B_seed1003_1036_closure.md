# YC random-weather 004_18B：seed1003 / seed1036 历史天气闭合审计

生成时间：2026-09-27T05:33:21.241800+00:00  
范围：只读历史训练记录；另对两个原 context 串行执行 no-op runtime horizon replay。未训练 PPO、未加载 checkpoint、未做正式评估。新 runtime crop execution（如有）仅为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，crop outcome 未使用。

## 结论

- `seed1003_status = UNVERIFIABLE`
- `seed1036_status = UNVERIFIABLE`
- `training_weather_history_status = PARTIALLY_VERIFIED_78_PLUS_2`
- 现有事实是两条历史 PPO episode 比 004_17 no-op candidate 多 2 天；但历史逐日状态和原动作缺失，**不能证明这是单纯 capture offset，也不能证明重叠逐日天气相同**。
- 两个历史整段 `runtime_weather_sha256` 与 004_17 no-op candidate 不同。由于 hash 覆盖整段序列且行数不同，hash mismatch 不能定位日值差异，不据此将状态定成 `MISMATCH`。
- 本轮同 context no-op replay 分别复现了 004_17 candidate 的 runtime hash（1003: `EC8FA734...FC1BDB`；1036: `AFCE2570...277DD3D`），表明当前 no-op capture 可重复；这不证明历史 PPO episode 的天气前缀或末尾两天相同。
- **若 004_19 要求宣称 100/100 历史 realization 全部闭合，目前不具备条件；若按敏感性并列审计 FULL80 candidate 与 EXACT78 verified subset，则可以有条件开展。** 不得把 FULL80 单独写成完全核验的历史训练域。

## Horizon 对照

| seed | crop-year | schedule episode | historical days | 004_17 no-op rows | 本轮 no-op steps | terminated / truncated | 最终 DAP / ISTAGE | 分类 |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1003 | 2011 | 62 | 110 | 108 | 108 | True / False | 103 / 6 | UNVERIFIABLE |
| 1036 | 2009 | 20 | 102 | 100 | 100 | True / False | 95 / 6 | UNVERIFIABLE |

字段与 hash 见 `termination_horizon_comparison.csv`、`seed1003_verification.json`、`seed1036_verification.json`。历史 episode 长度/hash 来自 004_05 多 PPO-seed training summaries；RSEED1、context、planting date 来自 schedule 与 004_17 verified manifest。

### 两日差异的解释边界

004_03 `ScheduledEpisodeEnv._step` 在环境返回 `terminated or truncated` 时记录 summary、`episode_days` 与天气 hash；004_17 的 `run_context_group` 同样在这两个 done flag 任一为真时停止 no-op capture。源码没有显示统一固定少采两日的额外切片。因此 capture horizon 差异不是已证实原因。原 PPO 管理动作与 no-op 不同，理论上可能改变 crop trajectory / maturity 与 episode 终止时点；但训练期没有逐日 action、final DAP/ISTAGE 或 done/truncated 记录，所以只能列作未验证机制。

本轮 replay 使用原 context schedule 的前序 seed 串行 reset，再对目标 seed 执行 action index 0（no irrigation/no N），只为补采当前 no-op 终止状态。它不是原历史动作 replay，也没有把任何 crop outcome 纳入分析。当前 `done_reason` 若 runtime `info` 未暴露，仅报告 terminated/truncated 标志，不猜测 harvest/maturity。

日期边界：runtime state 没有提供可用的真实 calendar date；004_17 legacy parser 使用 `2008-06-01 + state index` 作为 hash 日期标签。本轮末行标签 2008-09-16（seed1003）和 2008-09-08（seed1036）不是相应 2011/2009 crop-year 的真实结束日期，故真实终止日期记为 unavailable。

## 历史动作及逐日 overlap

- 004_05 可用的是 `training_episode_summary.csv` 汇总；没有目标历史 episode 的 per-step action trace。
- `evaluation_step_level_all_models.csv` 是 observed / held-out evaluation trace，不是训练 weather seeds 1003/1036 的原 training episode。没有用 evaluation action 替代训练动作。
- `historical_action_sequence_seed1003.csv` 和 `historical_action_sequence_seed1036.csv` 均标记 `ACTION_TRACE_NOT_AVAILABLE`。
- 两份 overlap CSV 均为 `NO_HISTORICAL_DAILY_TRACE`，可比较重叠天数为 0；RAIN/SRAD/TMAX/TMIN 的逐行误差因此为空，而不是 0。

## 科学状态与下一步约束

目前只能称 **78 条 exact + 2 条 unresolved/provisional**，不能称 80/80 exact。两条 candidate 可用于 descriptive sensitivity，但不能升级为 historical exact weather。004_18A 已显示排除 provisional realization 会改变部分观测年的 rainfall support，因此 004_19 若之后执行，应并列 FULL80/EXACT78 并显式标注 provisional；如果问题要求唯一正式历史训练域结论，则需先有两条原始历史 action/state/weather trace。此轮不启动 004_19。

## 用户问题逐项回答

1. 两条都少 2 天；源码未显示通用的固定两日 capture offset。历史 PPO 管理和 no-op 管理导致 episode 终止时点不同，是可能机制但未证实。
2. 差异发生在 episode horizon 上；两边都由 terminated/truncated 控制停止。历史 done flags 未保存，具体终止类型未知。
3. 原历史动作可能影响 episode 长度，但动作 trace 缺失，不能验证。
4. seed1003 未恢复为可核验的历史原天气，状态为 `UNVERIFIABLE`。
5. seed1036 未恢复为可核验的历史原天气，状态为 `UNVERIFIABLE`。
6. 当前 training weather 是 78 条 exact + 2 条 provisional/unresolved，不是 80/80 exact。
7. 可以有条件开展 004_19：必须并列 FULL80 candidate 与 EXACT78 verified subset，并将两条标成 provisional；若要作“80/80 历史天气均 exact”的单一正式结论则条件不满足。

## 证据索引

- schedule：`results/yc_random_weather_ppo/004_05/config/training_weather_schedule_random.csv`
- 训练汇总：`results/yc_random_weather_ppo/004_05/training/random_weather/ppo_seed_*/training_episode_summary.csv`
- evaluation-only step trace：`results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv`
- terminal semantics：`results/yc_random_weather_ppo/004_03/run_controlled_pilot.py` 的 `ScheduledEpisodeEnv._step`
- context replay：`src/run_004_17_yc_runtime_wgen_capture_and_weather_recovery.py` 的 `run_context_group`
- archived candidates：`results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/`
- replay error(s)：无
