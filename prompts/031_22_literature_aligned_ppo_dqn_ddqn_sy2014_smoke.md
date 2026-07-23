# 031_22 Literature-aligned PPO / DQN / Double-Dueling DQN comparison smoke

## Purpose

The current free-timing MaskablePPO branch has produced the best result among recent SY2014 free-timing experiments, but the literature contains both PPO and DQN-style approaches for crop water/nitrogen management.

This task runs a small, pre-registered algorithm comparison under identical SY2014 free-timing conditions:

- existing MaskablePPO seed0, 20k steps from 031_17;
- existing SB3 vanilla DQN seed0, 20k steps from 031_19;
- new project-local mask-aware Double-Dueling DQN seed0, 20k steps.

## Scope

- SYA2014 only.
- seed0 only.
- Same action grid as 031_17/031_19.
- Same reward as 031_17/031_19.
- Same action safety constraints as 031_17/031_19.
- No reward tuning.
- No site/year expansion.
- No post-hoc checkpoint selection; evaluate final model only.

## Why Double-Dueling DQN

Crop-management RL literature includes DQN variants, including Double DQN and Dueling DQN / Dueling DDQN, for discrete fertilization decisions. Since SB3 DQN does not natively support dynamic action masks, this task uses a project-local mask-aware implementation:

- exploration samples only legal actions;
- greedy action selection masks illegal actions;
- replay stores current and next action masks;
- Double-DQN target uses online network for masked next-action selection and target network for target Q evaluation;
- Dueling architecture estimates value and advantage separately.

This is not claimed to be a full reproduction of any single paper. It is a literature-aligned algorithm-family comparison.

## Fixed environment and reward

Action grid:

- irrigation levels: `[0, 6, 12, 18, 24]` mm
- nitrogen levels: `[0, 40, 80, 120, 160]` kg/ha

Safety:

- seasonal irrigation cap: 160 mm
- seasonal nitrogen cap: 250 kg/ha
- minimum irrigation interval: 7 days
- minimum fertilization interval: 7 days
- irrigation allowed DAP: 1-120
- fertilization allowed DAP: 1-90

Reward:

```text
reward = 0.001 * (0.158 * final_yield - 1.1 * irrigation - 1.58 * nitrogen)
```

where water/nitrogen costs are charged on action steps and final yield is credited at harvest.

## Success interpretation

This task does not decide the final thesis algorithm. It answers:

> Under the current free-timing discrete setup, does a mask-aware Double-Dueling DQN look competitive with the existing PPO seed0 result?

Branch guide:

- A: Double-Dueling DQN clearly exceeds PPO on yield/profit/PFP_N with similar or lower stress → consider DQN variant as next main candidate.
- B: Double-Dueling DQN improves over vanilla DQN but remains below PPO → PPO remains main branch, DQN is retained as literature control.
- C: Double-Dueling DQN is unstable or worse than vanilla DQN → stop DQN enhancement until a separate DQN diagnostic is approved.

