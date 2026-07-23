# 031_03 free-daily original-reward random/no-op baseline

## Purpose

Clarify the interpretation of 031_01/031_02.

031_01 PPO and 031_02 DQN were only 512-step smoke runs. They showed cap-saturated behavior, but that does not yet prove the algorithms learned a cap-saturating strategy. This baseline tests whether the same free-daily original-reward environment naturally reaches the caps under untrained/random action selection.

## Scope

- Site-year: SYA2014 only.
- No training.
- Same free daily decision setting as 031_01/031_02.
- No expert DAP windows.
- Minimum interval: 1 day.
- Fertilization latest DAP: 90.
- Season caps:
  - irrigation <= 160 mm
  - nitrogen <= 250 kg/ha
- Reward:

```text
reward_t = delta_GRNWT_t - 1.0 * irrigation_t - 5.0 * nitrogen_t
```

No terminal bonus, no `/1000`, no TOPWT term.

## Policies to evaluate

1. `continuous_noop`: PPO-style continuous action wrapper, always request zero water and zero nitrogen.
2. `continuous_random`: PPO-style continuous action wrapper, sample random continuous actions from the environment action space.
3. `dqn_noop`: DQN-style discrete wrapper, always action 0 = no-op.
4. `dqn_random`: DQN-style discrete wrapper, sample random discrete action from the 3×3 grid.

## Interpretation

- If random policies also hit I/N caps, 031_01/031_02 cap saturation is not enough evidence that PPO/DQN learned a cap-saturating strategy.
- If no-op policies do not hit caps and produce much lower yield, the action wrappers and safety caps are working.
- This task does not tune reward or train a model.
