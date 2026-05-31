# Reward Sweep Backup Notes

## Original reward parameters

The original fertilization reward parameters for maize are:

```python
"maize": {"coef": 1.0, "penality": 0.5}
```

The value `coef=1.5, penality=0.5` is not the original setting. It was only a candidate
selected from the observed PPO result plateau in the sweep outputs.

## Current reproducibility check

The local scripts `train_hl.py`, `evaluate_hl.py`, and `optimize_reward_hl.py` pass reward
parameters through environment variables:

```text
GYM_DSSAT_REWARD_COEF
GYM_DSSAT_REWARD_PENALITY
```

On 2026-05-31, the installed DSSAT reward file checked at:

```text
/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/rewards.py
```

still contained the original hard-coded reward function and did not read these environment
variables. Therefore, any final claim about the best `coef` and `penality` must confirm that
the reward patch was applied before the official rerun.

## Safe patch helper

Use `tools/patch_dssat_rewards.py` to patch the installed reward file with an automatic
backup. The helper keeps the original maize defaults as `coef=1.0, penality=0.5`, but allows
training and evaluation scripts to override them through environment variables.
