# 032_01 LC2010 rule-timing counterfactual audit

## Purpose

After 032_00, MaskablePPO no longer produced 6 mm micro-irrigation but still front-loaded N and irrigation in LC2010. This task checks whether that front-loading is favored by the LC2010 DSSAT/reward landscape itself, or whether the RL model failed to discover a better timing pattern.

## Scope

- Site-year: LCA2010 only.
- No PPO/DQN training.
- No parameter tuning.
- Use the same coarse action grid and safety constraints as 032_00.
- Run deterministic fixed-rule forward simulations.

## Rules to compare

All rules are deterministic and subject to the same action safety caps:

1. `early_frontload`: approximate the 032_00 MaskablePPO behavior, applying high N and moderate irrigation early.
2. `uniform_spread`: distribute water/N over several phenology-like DAPs without using official expert DAP as a hard policy.
3. `midseason_shift`: delay most water/N until mid-season.
4. `late_delayed`: intentionally late water/N, expected to be poor if timing matters.
5. `stress_triggered`: apply water/N only after SWFAC or NSTRES exceeds 0.05.
6. `ppo_03200_replay`: replay the exact 032_00 MaskablePPO action sequence.

## Metrics

For each rule:

- final grain yield;
- total irrigation;
- total nitrogen;
- simple unscaled return under 032_00 reward accounting;
- irrigation and fertilization event counts;
- first irrigation and first N DAP;
- W and N stress days above 0.05;
- action sequence;
- daily CSV path.

## Decision logic

- If `early_frontload` or `ppo_03200_replay` is near-best, the current environment/reward does not strongly penalize front-loading in LC2010; fixing it requires changing the decision/reward definition, not just more training.
- If `uniform_spread`, `midseason_shift`, or `stress_triggered` is clearly better, then PPO/DQN failed to learn a better timing strategy under the current training setup.
- This audit is descriptive/diagnostic; it does not select a new policy.

