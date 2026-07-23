# 031_10 DQN literature design audit before next free-timing smoke

## Purpose

Before changing the DQN design again, summarize what the directly relevant literature implies for our next free-timing DQN experiment.

This is a design audit, not a training run.

## Sources checked

1. Balderas et al. 2024/2025, *A comparative study of deep reinforcement learning for crop production management*.
   - Reports that PPO outperformed DQN in single fertilization and irrigation tasks, but DQN performed better than PPO in the mixed water-nitrogen task.
   - Also notes DQN had more reward fluctuation and was more sensitive to parameter choices.
   - In mixed management, DQN applied N mainly around days 60-80 and irrigation around days 80-120, while PPO behaved poorly in that setting.

2. Wu et al. 2022, *Optimizing Nitrogen Management with Deep Reinforcement Learning and Crop Simulations*.
   - Uses DSSAT/gym-DSSAT daily interaction with DQN and SAC.
   - Supports DQN as a legitimate crop-management baseline, especially where actions are discretized.

3. Tao et al. 2023, *Optimizing Crop Management with Reinforcement Learning and Imitation Learning*.
   - Uses DQN for full-observation DSSAT crop management, then imitation learning for partial observation.
   - Reinforces that DQN may be useful as a high-information simulator policy, but deployable/reduced-observation policies require additional treatment.

4. Gautron et al. 2022, *gym-DSSAT: a crop model turned into a Reinforcement Learning environment*.
   - Defines daily fertilization, irrigation, and mixed management tasks in gym-DSSAT.
   - Notes that crop RL needs tailoring; untuned algorithms can work in some cases but are not a complete recipe.

## What this means for our project

### 1. DQN is literature-supported, but not automatically superior

The comparative study supports our motivation to keep a DQN branch, especially for mixed water-nitrogen management. However, the same paper also reports DQN instability and parameter sensitivity. Therefore, DQN should be a pre-registered comparator or candidate branch, not an assumed replacement for PPO.

### 2. Our 031_09 result is consistent with the literature pattern

031_09 literature-aligned DQN did better than the free-timing PPO/random early-dump baselines on SY2014:

- It did not reach both water and nitrogen caps by DAP10.
- It achieved 10794.46 kg/ha, better than free-timing PPO/random, but still below the fixed-timing reference around 10908.60 kg/ha.
- It applied N early and mid-early, then irrigated repeatedly around DAP43-49.

This is directionally similar to the comparative paper's observation that DQN can produce more balanced mixed-management behavior than PPO, but our policy still saturates caps and shows action-clipping aliasing.

### 3. Next DQN change should target action realism, not just network size

The biggest 031_09 pathology is not obviously the MLP architecture. It is action execution:

- DQN repeatedly requested action 19, raw I18/N160.
- After the seasonal N cap was reached, that same requested action was clipped to I18/N0.
- This creates action aliasing: the Q-network thinks it is choosing one action, but the environment executes another.

Therefore, the next DQN smoke should first reduce this aliasing by adding a minimum operation interval or by making the action mask budget-aware.

### 4. Literature-consistent DQN design choices for next smoke

Keep:

- discrete action DQN;
- replay buffer;
- target network;
- epsilon-greedy exploration;
- 3-layer MLP with 256 hidden units as the literature-aligned architecture;
- 25-action water-nitrogen grid unless a pre-registered action-grid reduction is explicitly tested.

Change only one thing next:

- add a hard minimum interval mask, preferably 7 days, for same-resource operations.

Do not simultaneously change:

- reward weights;
- training length;
- network size;
- action grid;
- site/year;
- seed count.

## Recommended next task

031_10 should be:

**Literature-aligned DQN + free daily timing + 7-day same-resource interval hard mask, SY2014 seed0, 5k timesteps.**

Compare against:

- 031_09 literature-aligned DQN without interval;
- 031_08 PPO reward-v3;
- 031_03 random;
- 031_05 uniform/expert-window rule references.

Success criteria:

1. no DAP1-DAP10 cap saturation;
2. fewer clipped/aliased repeated N160 requests than 031_09;
3. yield at least close to 031_09, preferably closer to 10908.60;
4. operations are more separated in time;
5. no expansion to all sites unless this single smoke passes.

