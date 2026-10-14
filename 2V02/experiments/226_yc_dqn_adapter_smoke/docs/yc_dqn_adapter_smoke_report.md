# YC DQN adapter smoke report

审计日期：2026-09-11

本轮仅进行工程 smoke，不评价产量、不调参、不执行正式 3-seed 训练、不进行 PPO-vs-DQN 结果比较。现有正式 PPO/029/032/055 文件未被覆盖。

## Attempt history

### attempt_01 — staged irrigation assertion failure

- 时间：`2026-09-11 17:34:19`（Asia/Shanghai；依据归档失败 CSV/JSON 的文件 `LastWriteTime`，不是独立的进程事件时间；精确进程 wall-clock 时间为 `unknown / insufficient evidence`）。
- 阶段：静态 staged-irrigation assertion，发生在真实 YC environment 初始化、reward parity、transition collection 和 gradient update 之前；失败点为 `run_staged_assertions`。
- 具体原因：测试 probe 的临时 `ACTION_GRID` 使用了 `irrigation_mm/nitrogen_kg_ha` 字段名，而生产 wrapper 的 action legality 函数读取 `amir/anfer`。因此 probe 将正灌溉 candidate 读成了 `0`，导致 DAP30/60/90/season-cap 的正灌溉边界出现 `expected_valid=False` 但 `actual_valid=True`。这是 adapter smoke 测试夹具字段未对齐，不是正式 wrapper 的 staged constraint 失败。
- 失败证据：`results/attempt_01/yc_dqn_smoke_summary.json`（`AssertionError: YC staged irrigation assertions failed`）和 `results/attempt_01/yc_dqn_staged_irrigation_assertions.csv`（四个正灌溉边界行记录为 FAIL）。

### From attempt_01 to final PASS

1. 将 smoke runner 内部 action-grid probe 的字段改为生产 wrapper 使用的 `amir/anfer`，并同步修正 mapping/staged/N assertion 的读取字段。
2. 后续 QA 发现原始 observation 的 `observation_variables` 含有 6 维 `sw` 向量，而实际 flattened observation 是 22 维；因此只补充了 observation provenance logging，将 `sw` 展开记录为 `sw[0]...sw[5]`，使字段顺序记录与 22 维输入一致。该记录修正的中间版本保存在 `results/attempt_02_pre_field_expansion/`。
3. 以上均为 adapter smoke 的工程修复/证据记录修复，不是科学契约修改。

### Contract-change audit

从 `attempt_01` 到最终 PASS 版本，以下项目均未改变：

| 项目 | 是否改变 | 说明 |
|---|---|---|
| observation | 否 | 仍使用当前 YC 055-style raw daily observation；只补充 flattened `sw[0..5]` 的记录。 |
| reward | 否 | 仍直接调用当前 `LateIrrigationReserveMaskWrapper` 的 040_28/040_36 chain；未复制或调节 reward。 |
| action space | 否 | 仍为 I `[0,15,30,45]` × N `[0,40,80,120]` 的 16-action row-major mapping；只修正 probe 字段名。 |
| mask logic | 否 | 仍复用 029 mask-aware core 和当前 PPO wrapper；未改变 exploration/greedy/Bellman/replay mask 逻辑。 |
| irrigation/fertilizer constraints | 否 | staged I 75/150/195/240、late reserve 45、N cap 250、daily cap、7-day interval 和 DAP windows 均未改变。 |
| training years | 否 | formal train/validation contract 仍为 2004–2013 / 2014–2023；smoke year 仍为 YC 2004。 |
| IC | 否 | 仍为 `lowIC`。 |
| cultivar | 否 | 仍为当前 YC calibrated cultivar / `MZCER048.CUL`。 |
| DQN hyperparameters | 否 | 仍为报告中标记的 `engineering_smoke_only; not_for_formal_comparison` 配置；没有根据失败或 smoke reward 调参。 |

失败 attempt 已保留，未删除或覆盖：

- `experiments/226_yc_dqn_adapter_smoke/results/attempt_01/yc_dqn_smoke_summary.json`
- `experiments/226_yc_dqn_adapter_smoke/results/attempt_01/yc_dqn_staged_irrigation_assertions.csv`
- `experiments/226_yc_dqn_adapter_smoke/results/attempt_02_pre_field_expansion/yc_dqn_adapter_smoke_report.md`
- `experiments/226_yc_dqn_adapter_smoke/results/attempt_02_pre_field_expansion/yc_dqn_observation_contract.csv`

## 最终结论：`READY_FOR_FORMAL_CONFIG_FREEZE`

`Proceed to formal DQN config freeze: YES`

## Contract freeze

- Site/site code: `YCA/YC`；input profile: `lowIC`；training smoke year: `2004`。
- Formal split retained: train `2004–2013`, validation `2014–2023`。
- Observation: current raw daily observation, no normalization, no forecast; DQN receives the same vector from the current PPO wrapper. 
- Actions: I `[0,15,30,45]` × N `[0,40,80,120]`, 16 joint actions, row-major. Mapping is in `results/yc_dqn_adapter_action_mapping.csv`. 
- Safety/reward: the adapter calls `LateIrrigationReserveMaskWrapper` directly, preserving the current 040_28/040_36 chain; it does not duplicate reward logic.

## Gate results

- exploration_mask_safe: `PASS`
- greedy_mask_safe: `PASS`
- bellman_target_mask_safe: `PASS`
- replay_action_effect_consistency: `PASS`
- staged irrigation assertions: `PASS`
- N constraints: `PASS`
- observation contract: `PASS`
- reward parity: `PASS`

## Short smoke closure

- env_steps: `12`
- optimizer_updates: `9`
- target_sync_count: `3`
- invalid_action_attempts: `0`
- fallback_count: `0`
- replay_size: `12`
- checkpoint_saved: `True`
- checkpoint_reload_success: `True`
- deterministic eval steps: `8`
- deterministic eval constraints: `True`

## 对 prompt 的 13 个问题逐项回答

1. **029 core 是否接入当前 YC daily environment？** `PASS`；适配器使用当前 YC lowIC daily wrapper，DQN core 只负责 mask-aware action/replay/update。
2. **observation 是否与 PPO 完全一致？** `PASS`；记录见 `results/yc_dqn_observation_contract.csv`。
3. **reward 是否与 PPO 完全一致？** `PASS`；DQN adapter 不重写 reward，parity 见 `results/yc_dqn_reward_parity_check.csv`。
4. **16-action mapping 是否完全一致？** `PASS`；唯一 mapping 见 `results/yc_dqn_adapter_action_mapping.csv`。
5. **exploration 是否 mask-safe？** `PASS`。
6. **greedy 是否 mask-safe？** `PASS`。
7. **Bellman target 是否 mask-safe？** `PASS`。
8. **replay action 是否与真实 executed action 一致？** `PASS`；逐 transition 记录见 `results/yc_dqn_transition_consistency.csv`。
9. **staged irrigation 75/150/195/240 + late reserve 45 是否全部 PASS？** `PASS`。
10. **N constraints 是否全部 PASS？** `PASS`。
11. **collection/update/target-sync/save/reload/eval 是否闭环？** `PASS`。
12. **是否出现 invalid action/fallback？** `invalid_action_attempts=0, fallback_count=0`。
13. **readiness？** `READY_FOR_FORMAL_CONFIG_FREEZE`。

## Smoke-only hyperparameters

- label: `engineering_smoke_only; not_for_formal_comparison`。
- seed `0`; steps `12`; replay capacity `64`; batch size `4`; learning rate `3e-4`; gamma `1.0`; CPU; MLP `obs_dim -> 32 -> 32 -> 16`; gradient clip `10`; target sync every `4` steps。
- 这些数值不构成 formal DQN freeze，也没有根据 reward/yield 进行调整。

## Outputs

- `results/yc_dqn_staged_irrigation_assertions.csv`
- `results/yc_dqn_n_constraint_assertions.csv`
- `results/yc_dqn_observation_contract.csv`
- `results/yc_dqn_reward_parity_check.csv`
- `results/yc_dqn_transition_consistency.csv`
- `results/yc_dqn_smoke_summary.csv`
- `results/yc_dqn_smoke_train_daily.csv`
- `results/yc_dqn_smoke_eval_daily.csv`
- `checkpoints/yc_dqn_checkpoint_000012.pt`
