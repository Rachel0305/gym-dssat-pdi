# Current reward function review

Source reviewed: `Not directly readable from this host path; see docs/all_mode_reward_and_observation_notes.md.`

## Code or project note excerpt

```python
# All-mode reward and observation notes

## Which reward file is actually used

Training imports the installed package inside the Docker container:

```text
/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/rewards.py
```

Windows may expose the same container filesystem through an overlay path like:

```text
\\wsl.localhost\docker-desktop\mnt\docker-desktop-disk\data\desktop-containerd\daemon\io.containerd.snapshotter.v1.overlayfs\snapshots\102\fs\opt\gym_dssat_pdi\lib\python3.10\site-packages\gym_dssat_pdi\envs\configs\rewards.py
```

Python does not import the UNC path directly. It imports the container path above. The patch
helper `tools/patch_dssat_rewards.py` modifies that installed file inside the container.

## All-mode reward call chain

`train_hl_all.py` creates the environment with:

```python
env_args = {
    'mode': 'all',
    ...
}
```

The installed environment calls `get_reward_function(mode)` in the installed `rewards.py`.
For `mode='all'`, it returns `all_reward`.

`train_hl_all.py` and `evaluate_hl_all.py` pass the sweep parameters through environment
variables before creating the DSSAT environment:

```text
GYM_DSSAT_REWARD_COEF
GYM_DSSAT_REWARD_PENALITY
GYM_DSSAT_ALL_FERT_WEIGHT
GYM_DSSAT_ALL_IRRIG_WEIGHT
GYM_DSSAT_ALL_ANFER_COST
GYM_DSSAT_ALL_AMIR_COST
GYM_DSSAT_ALL_ANFER_EXCESS_LIMIT
GYM_DSSAT_ALL_AMIR_EXCESS_LIMIT
GYM_DSSAT_ALL_ANFER_EXCESS_COST
GYM_DSSAT_ALL_AMIR_EXCESS_COST
```

The patched `all_reward` computes:

```text
fert_weight * fertilization_reward
+ irrig_weight * irrigation_reward
- extra action costs
- excess total water/nitrogen costs
```

## Where trnu, nstres, and swfac come from

The DSSAT environment returns a dictionary observation. `GymDssatWrapper` converts that
dictionary into a NumPy vector for Stable-Baselines3 using the environment's
`observation_variables` order.

During evaluation, `evaluate_hl_all.py` reconstructs a dictionary from the vector:

```python
obs_keys = env.unwrapped.observation_variables
obs_dict = dict(zip(obs_keys, observation))
```

`nstres` and `swfac` are read directly from `obs_dict`:

```python
'swfac': obs_dict.get('swfac', np.nan)
'nstres': obs_dict.get('nstres', np.nan)
```

`trnu` is not always part of the vector observation, so evaluation looks for it in multiple
places:

```python
latest_obs = get_latest_observation_dict(env, info=info, obs_dict=obs_dict)
trnu = safe_float(latest_obs.get('trnu', np.nan))
sync_trnu_to_history(env, trnu)
```

`get_latest_observation_dict` merges the latest raw DSSAT history observation, environment
state-like attributes, `info`, and `obs_dict`. Then `sync_trnu_to_history` writes the TRNU
value back into `env.unwrapped.history['observation'][-1]` so plotting and pickle histories
also contain TRNU.

```

## Diagnosis

- 当前 all-mode reward 主要来自产量/生长信号与水氮动作成本的组合。
- 从 HLA 2007 debug PPO 的 daily output 看，策略几乎每天输出正的 amir/anfer，说明当前训练仅靠 reward 还不足以约束动作频率。
- 如果 action cost 或 season excess penalty 不够强，PPO 在短步数随机探索阶段会频繁尝试高水高氮动作。
- 当前结果与 fixed_high_input 的 120 mm / 250 kg ha-1 对照不一致，说明不能直接批量训练。
- 本阶段不覆盖 reward；先用 action safety 单独隔离动作尺度问题。
