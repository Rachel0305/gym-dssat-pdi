# 040_46_sya_lowIC_04040_policy_observation_audit

## 结论

- `observation_names_unresolved_run_runtime_probe`

## 边界

- 不训练。
- 不修改 reward。
- 不修改动作、安全层或 action mask。
- 只检查 040_40 MaskablePPO 实际接收的 observation。

## 汇总

| station_code   |   year | observation_shape   |   observation_size |   observation_names_count | observation_names_source           | names_match_vector   | explicit_dap_in_named_policy_observation   |   dap_index | latest_observation_dict_has_dap   |   latest_observation_dict_dap | named_time_proxy_variables                                                              |   action_mask_size |   valid_action_count_at_reset | observation_space_repr        | action_space_repr   |
|:---------------|-------:|:--------------------|-------------------:|--------------------------:|:-----------------------------------|:---------------------|:-------------------------------------------|------------:|:----------------------------------|------------------------------:|:----------------------------------------------------------------------------------------|-------------------:|------------------------------:|:------------------------------|:--------------------|
| SYA            |   2005 | [25]                |                 25 |                        17 | env_chain[0].observation_variables | False                | False                                      |           1 | True                              |                             0 | ['dap', 'dtt', 'grnwt', 'istage', 'rtdep', 'topwt', 'totir', 'vstage', 'wtdep', 'xlai'] |                  9 |                             9 | Box(0.0, inf, (25,), float32) | Discrete(9)         |
| SYA            |   2014 | [25]                |                 25 |                        17 | env_chain[0].observation_variables | False                | False                                      |           1 | True                              |                             0 | ['dap', 'dtt', 'grnwt', 'istage', 'rtdep', 'topwt', 'totir', 'vstage', 'wtdep', 'xlai'] |                  9 |                             9 | Box(0.0, inf, (25,), float32) | Discrete(9)         |

## 输出

- `benchmark_results/040_46_sya_lowIC_04040_policy_observation_audit/tables/040_46_policy_observation_summary.csv`
- `benchmark_results/040_46_sya_lowIC_04040_policy_observation_audit/tables/040_46_policy_observation_variables.csv`
