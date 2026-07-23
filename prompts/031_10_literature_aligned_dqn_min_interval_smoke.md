# 031_10 Literature-aligned DQN with minimum operation interval

## Rationale

031_09 showed that a literature-aligned DQN reconstruction outperformed the free-timing PPO/random early-dump baselines on SY2014, but still saturated I160/N250 and repeatedly requested clipped high-N actions.

Relevant crop-management RL literature supports DQN as a legitimate discrete-action baseline for DSSAT/gym-DSSAT, especially in mixed water-nitrogen tasks, but also reports higher variability and parameter sensitivity. Therefore, 031_10 should not broadly tune DQN. It should test one concrete pathology from 031_09: unrealistic repeated operations and action clipping.

## Experiment

- Site-year: SYA2014.
- Seed: 0.
- Algorithm: literature-aligned DQN reconstruction.
- Training budget: 5,000 timesteps.
- Decision timing: daily free timing.
- Expert DAP windows: not used.
- New single change from 031_09:
  - same-resource minimum operation interval = 7 days.
  - irrigation after an irrigation event is masked/clipped to zero until 7 days have passed.
  - fertilization after a fertilization event is masked/clipped to zero until 7 days have passed.
- Keep all other 031_09 choices fixed:
  - 25-action water-nitrogen grid.
  - 3 x 256 MLP.
  - replay buffer, target network, epsilon-greedy settings.
  - literature reward weights.
  - seasonal caps I160/N250.
  - DAP > 90 N disabled.

## Success checks

1. DAP1-DAP10 does not reach both I160 and N250.
2. Repeated clipped N160 requests are reduced relative to 031_09.
3. Yield does not collapse relative to 031_09.
4. Yield moves toward the fixed-timing reference, if possible.
5. Daily output and nonzero action table are saved.

## Stop rule

This remains a single-site-year smoke. Even if successful, do not expand directly to all sites/years without a follow-up seed check.

