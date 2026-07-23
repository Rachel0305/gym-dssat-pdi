# 031_02 free-daily original-reward DQN smoke

## Purpose

Test whether switching the free-daily original-reward smoke from PPO to DQN changes the early behavior observed in 031_01.

This is a small diagnostic smoke, not a full multi-year experiment.

## Scope

- Site-year: SYA2014 only.
- Seed: 0 only.
- Training budget: 512 timesteps.
- No expert DAP windows.
- No 7-day minimum interval.
- Fertilization allowed only before or at DAP 90.
- Season caps unchanged from 031_01:
  - irrigation <= 160 mm
  - nitrogen <= 250 kg/ha
- Reward unchanged from 031_01:

```text
reward_t = delta_GRNWT_t - 1.0 * irrigation_t - 5.0 * nitrogen_t
```

No terminal feasibility bonus, no `/1000` reward scaling, no TOPWT term.

## Required DQN discretization

DQN requires a discrete action space, so this smoke is not a mathematically pure algorithm-only swap from continuous-action PPO. The discrete action grid must be explicit:

| action_index | irrigation_mm | nitrogen_kg_ha |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 40 |
| 2 | 0 | 80 |
| 3 | 20 | 0 |
| 4 | 20 | 40 |
| 5 | 20 | 80 |
| 6 | 40 | 0 |
| 7 | 40 | 40 |
| 8 | 40 | 80 |

The action safety layer still clips by DAP range, season caps, and daily caps.

## Pre-run checks

1. Run inside Docker container `nifty_taussig`.
2. Use `/opt/gym_dssat_pdi/bin/python`.
3. Confirm `stable_baselines3.DQN` imports in that environment.
4. Write prompt, config, script, CSV/JSON/MD record before reporting.

## Expected outputs

- `benchmark_results/031_02_free_daily_original_reward_dqn_smoke/031_02_result.json`
- `benchmark_results/031_02_free_daily_original_reward_dqn_smoke/031_02_free_daily_original_reward_dqn_smoke_record.md`
- training summary CSV
- evaluation summary CSV
- daily output CSV

## Interpretation

Compare with 031_01 PPO smoke:

- whether DQN also saturates I160/N250 rapidly;
- whether DQN obtains higher/lower final GRNWT;
- whether DQN produces fewer continuous early operations;
- whether the learned action sequence looks agronomically more plausible.

Do not expand to all years/sites unless this smoke is technically valid and the user explicitly approves.
