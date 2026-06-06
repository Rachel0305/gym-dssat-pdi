# Current reward review

Generated at: 2026-06-06

## Reward source

- File path: `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/rewards.py`
- Mode used by PPO: `all`
- Functions: `all_reward`, `fertilization_reward`, `irrigation_reward`

## Code snippets

```python
def fertilization_reward(_previous_state, _next_state, _history, _cultivar):
    weights = {
            "maize"  : {"coef":1.0, "penality":0.5},
            "cotton" : {"coef":1.0, "penality":0.75},
            "rice"   : {"coef":1.0, "penality":0.5},
    }
    if _next_state:
        last_action = _history['action'][-1]['anfer']
        cultivar_weights = weights[_cultivar]
        penality = _reward_float(
            'GYM_DSSAT_REWARD_PENALITY',
            _reward_float('GYM_DSSAT_REWARD_PENALTY', cultivar_weights["penality"])
        )
        coef = _reward_float('GYM_DSSAT_REWARD_COEF', cultivar_weights["coef"])
        trnu = _next_state['trnu']
        return trnu * coef - penality * last_action
    return None


def irrigation_reward(_previous_state, _next_state, _history, _cultivar):
    if _next_state:
        last_action = _history['action'][-1]['amir']
        previous_topwt = _previous_state['topwt']
        next_topwt = _next_state['topwt']
        penality = 15
        return next_topwt - previous_topwt - penality * last_action
    return None


def all_reward(_previous_state, _next_state, _history, _cultivar):
    ferti_reward_value = fertilization_reward(_previous_state, _next_state, _history, _cultivar)
    irrig_reward_value = irrigation_reward(_previous_state, _next_state, _history, _cultivar)
    if ferti_reward_value is None or irrig_reward_value is None:
        return None
    fert_weight = _reward_float('GYM_DSSAT_ALL_FERT_WEIGHT', 1.0)
    irrig_weight = _reward_float('GYM_DSSAT_ALL_IRRIG_WEIGHT', 1.0)
    action_history = _history.get('action', [])
    last_action = action_history[-1] if action_history else {}
    last_anfer = float(last_action.get('anfer', 0.0))
    last_amir = float(last_action.get('amir', 0.0))
    total_anfer = sum(float(action.get('anfer', 0.0)) for action in action_history)
    total_amir = sum(float(action.get('amir', 0.0)) for action in action_history)

    extra_anfer_cost = _reward_float('GYM_DSSAT_ALL_ANFER_COST', 0.0)
    extra_amir_cost = _reward_float('GYM_DSSAT_ALL_AMIR_COST', 0.0)
    no_stress_amir_cost = _reward_float('GYM_DSSAT_ALL_AMIR_NO_STRESS_COST', 0.0)
    water_stress_threshold = _reward_float('GYM_DSSAT_ALL_WATER_STRESS_THRESHOLD', 0.05)
    excess_anfer_limit = _reward_float('GYM_DSSAT_ALL_ANFER_EXCESS_LIMIT', 1e12)
    excess_amir_limit = _reward_float('GYM_DSSAT_ALL_AMIR_EXCESS_LIMIT', 1e12)
    excess_anfer_cost = _reward_float('GYM_DSSAT_ALL_ANFER_EXCESS_COST', 0.0)
    excess_amir_cost = _reward_float('GYM_DSSAT_ALL_AMIR_EXCESS_COST', 0.0)

    swfac = float(_next_state.get('swfac', 0.0))
    no_stress_irrig_cost = no_stress_amir_cost * last_amir if swfac <= water_stress_threshold else 0.0
    action_cost = extra_anfer_cost * last_anfer + extra_amir_cost * last_amir + no_stress_irrig_cost
    excess_cost = (
        excess_anfer_cost * max(0.0, total_anfer - excess_anfer_limit)
        + excess_amir_cost * max(0.0, total_amir - excess_amir_limit)
    )
    return fert_weight * ferti_reward_value + irrig_weight * irrig_reward_value - action_cost - excess_cost

```

## Inputs and state variables

- Inputs: previous state, next state, history, cultivar.
- Crop state variables used directly: `trnu`, `topwt`, `swfac`.
- Action variables used directly: `amir`, `anfer`.
- Cumulative actions are available through `history['action']`.

## Cost structure

- Daily irrigation cost: yes, through `irrigation_reward = delta_topwt - 15 * amir`.
- Daily nitrogen cost: yes, through `fertilization_reward = trnu * coef - penality * anfer`.
- Season irrigation cost: optional env-var hook only, default 0.
- Season nitrogen cost: optional env-var hook only, default 0.
- Nonlinear excess penalty: no, only optional linear excess hooks with default 0.
- Terminal grain-yield benefit: no explicit terminal reward.

## Main problem

The current reward gives positive growth/nitrogen-recovery incentives every step while action safety caps total water and nitrogen externally. In the 006_02 season-cap sensitivity analysis, PPO filled every cap level from 100/150 to 300/450. This indicates that the current reward does not make PPO internalize seasonal water and nitrogen scarcity; the wrapper prevents unsafe totals, but the reward still rewards using the entire allowed budget.
