# 031_19 Fair DQN smoke under 031_17 free-timing discrete scaled-reward setup

## Purpose

031_17/031_18 showed that free-timing discrete MaskablePPO with strict action masks and reward scaling (`reward_scale=0.001`) is the strongest current free-timing RL branch on SYA2014.

031_19 tests whether DQN can match or exceed that result under the same management setup.

## Scope

- Site-year: SYA2014 only.
- Seed: 0 only.
- Training length: 20,000 timesteps.
- Algorithm: SB3 DQN.
- Daily free timing; no expert DAP windows.
- Same discrete action grid, resource caps, timing restrictions, and scaled reward as 031_17.
- No reward coefficient change.
- No expansion to other seeds/site-years unless this seed0 smoke is competitive.

## Important DQN masking limitation

SB3 DQN does not natively support MaskablePPO-style action masks.

Therefore this experiment uses the following approximation:

- during training, if DQN selects an illegal action, the wrapper forces it to no-op and records `mask_forced_noop=True`;
- during deterministic evaluation, DQN uses masked greedy Q selection: Q values are computed for all actions, illegal actions are assigned `-inf`, and the legal action with maximum Q is selected.

This is the fairest lightweight DQN comparison available without implementing a custom masked-DQN algorithm.

## Fixed configuration

Reward:

```text
unscaled non-terminal = -1.1 * irrigation - 1.58 * nitrogen
unscaled terminal     = 0.158 * final_grnwt - 1.1 * irrigation - 1.58 * nitrogen
training reward       = 0.001 * unscaled reward
```

Action grid:

- irrigation: `[0, 6, 12, 18, 24]` mm
- nitrogen: `[0, 40, 80, 120, 160]` kg/ha

Constraints:

- no expert DAP windows;
- no-op always valid;
- season irrigation cap: 160 mm;
- season nitrogen cap: 250 kg/ha;
- irrigation allowed DAP 1-120;
- nitrogen allowed DAP 1-90;
- same-resource minimum interval: 7 days;
- actions requiring clipping are treated as illegal.

## DQN hyperparameters

Use the literature-aligned DQN settings from 031_15 unless otherwise stated:

- learning rate: 1e-5
- buffer size: 100000
- learning starts: 1024
- batch size: 1024
- gamma: 0.99
- train frequency: 4
- gradient steps: 1
- target update interval: 120
- exploration fraction: 1.0
- initial epsilon: 1.0
- final epsilon: 0.0
- max gradient norm: 10
- network: `[256, 256, 256]`
- weight decay: 0.001

## Reference results

031_17/031_18 scaled-reward discrete MaskablePPO:

- seed0: Y=10943.10, I=108, N=240, profit=9635.10
- seed1: Y=10973.43, I=96, N=240, profit=9677.43
- seed2: Y=10937.06, I=156, N=240, profit=9581.06

031_12 staged N200 counterfactual:

- Y=10882.18, I=72, N=200, profit=9810.18

## Branch rules

- A: If DQN seed0 is competitive with or better than PPO seed0 on yield/profit/resource economy, run seed1/seed2 under the same fixed DQN setup.
- B: If DQN seed0 reaches high yield but still clearly overuses resources or front-loads actions worse than PPO, record as partial but not superior.
- C: If DQN seed0 is clearly worse than PPO seed0, stop this DQN fairness branch unless the user explicitly authorizes a 100k DQN repeat.

Do not tune DQN hyperparameters after seeing this seed0 result.

## Required outputs

- `docs/031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0_record.md`
- copied record under `benchmark_results/031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0/`
- training/evaluation summary CSV
- daily train/eval CSVs
- resolved config copy
- model file

