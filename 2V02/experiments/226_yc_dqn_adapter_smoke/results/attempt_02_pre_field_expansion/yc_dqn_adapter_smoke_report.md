# YC DQN adapter smoke report

审计日期：2026-09-11

本轮仅进行工程 smoke，不评价产量、不调参、不执行正式 3-seed 训练、不进行 PPO-vs-DQN 结果比较。现有正式 PPO/029/032/055 文件未被覆盖。

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
