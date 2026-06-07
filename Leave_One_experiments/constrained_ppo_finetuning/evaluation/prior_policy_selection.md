# Prior policy selection

Generated at: 2026-06-06

`BC_constant_schedule_baseline` is not selected as the main learned prior because it is a deterministic expert replay baseline rather than a learned generalizable policy. It is useful as a reference but not as the primary PPO initialization target.

`BC_two_stage_classifier_regressor` is selected as the main prior because it handles sparse expert actions explicitly and passed the 006_09 gate: low irrigation, about 154 kg/ha N, and acceptable yield. `BC_random_forest_regressor` is kept as a backup because it also avoided cap saturation but had lower yield than the two-stage prior. `BC_mlp_regressor` is not recommended because it under-applied N and lost too much yield.

Prior limitation: selected expert schedules have zero irrigation and sparse nitrogen event actions, so constrained PPO is mainly learning around a nitrogen schedule prior rather than a full irrigation optimization prior.

| policy_name | mean_yield | mean_irrigation | mean_n | mean_yield_loss_vs_ppo | ok |
| --- | --- | --- | --- | --- | --- |
| BC_constant_schedule_baseline | 8225.2271 | 0.0 | 150.0 | -0.1962 | 10 |
| BC_mlp_regressor | 2059.4187 | 0.0 | 0.0 | 0.7005 | 10 |
| BC_random_forest_regressor | 6115.2526 | 0.0 | 85.8948 | 0.1106 | 10 |
| BC_two_stage_classifier_regressor | 8091.8537 | 0.0 | 153.575 | -0.1768 | 10 |
