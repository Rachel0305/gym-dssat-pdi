# 032_13 LC future-year policy input sensitivity audit

## Purpose

Audit why the frozen LC multi-year no-forecast MaskablePPO policy selected the same deterministic action sequence on LC2011-LC2020 in 032_12.

This task is diagnostic only:

- no training;
- no DSSAT baseline rerun;
- no model selection;
- no reward/constraint change.

## Source evidence

- Source model: `benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt75000.zip`
- Previous transfer record: `docs/032_12_lc_multiyear_75k_future_year_transfer_record.md`
- Previous result: LC2011-LC2020 all used `DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80`.

## Questions

1. Are raw weather variables (`rain`, `srad`, `tmax`, `tmin`) included in the actual PPO observation vector?
2. At the decision days DAP1, DAP8, and DAP15, how different are the PPO observation vectors across LC2011-LC2020?
3. Does the policy assign the same top action because action masks force it, or because the policy distribution strongly prefers it?
4. If weather differs across years but the policy still selects the same action, is that because weather is not an input, because early-season state variables are similar, or because the deterministic argmax is insensitive to those differences?

## Method

For each LC target year 2011-2020:

1. Rebuild the same 032_12 evaluation environment.
2. Load the frozen 75k MaskablePPO checkpoint.
3. Run one deterministic evaluation trajectory.
4. Immediately before the action at DAP1, DAP8, and DAP15, save:
   - raw observation vector;
   - observation variable names when available;
   - observation dictionary from the environment helper;
   - current action mask;
   - deterministic predicted action;
   - masked action probabilities when extractable from the policy;
   - current-day weather values used only for auditing/plotting.
5. Compute across-year observation distances at each target DAP.
6. Write a record that separates confirmed facts from interpretation.

## Output files

Write outputs under:

`benchmark_results/032_13_lc_future_year_policy_input_sensitivity_audit/`

Required tables:

- `tables/032_13_decision_state_action_audit.csv`
- `tables/032_13_observation_values_by_year_dap.csv`
- `tables/032_13_action_probabilities_by_year_dap.csv`
- `tables/032_13_weather_vs_observation_field_check.csv`
- `tables/032_13_pairwise_obs_distance_by_dap.csv`
- `tables/032_13_obs_weather_correlation_by_dap.csv`

Required record:

- `docs/032_13_lc_future_year_policy_input_sensitivity_audit_record.md`

## Interpretation rules

- If true observation variable names are unavailable, do not claim weather is absent merely from missing names. Report it as an observation-schema limitation.
- If observation vectors differ but the top action and probability margin are similar, state that the deterministic policy is insensitive to those early differences.
- If masks leave several legal actions but the policy still selects the same one, do not attribute identical actions to masking.
- Do not claim the fixed action sequence is agronomically optimal from this audit alone; this audit only explains the model-input/action mechanism.
