# 034 handoff: DSSAT external action linkage fix and current evidence

This note summarizes the recent 034-series work so another AI/reviewer can
quickly understand what changed, what is proven, and what is not yet solved.

## Executive summary

We found and fixed a critical interface issue between the Python RL policy and
DSSAT management execution.

Before the fix, the Python side could log non-zero irrigation/fertilizer
actions, but DSSAT could still ignore them if the rendered management section
used automatic/reported management modes (`IRRIG=R`, `FERTI=R`). After the fix,
dynamic RL runs render treatment 1 as linked management (`IRRIG=L`, `FERTI=L`),
so Python external actions are actually accepted by DSSAT.

The fix is verified across five station templates by comparing:

- Python safe requested totals.
- DSSAT `Summary.OUT` executed irrigation and nitrogen totals.
- Rendered `OVERVIEW.OUT` management mode.

The interface is now connected, but the current free-timing PPO reward/constraint
configuration is not yet agronomically successful on the FQA2014 50K test.

## Key files

Core code changes:

- `src/ppo_safe_rendering.py`
  - Added `set_management_modes_for_treatment(...)`.
  - Added `linked_management` option to `build_env_args(...)`.
  - When `linked_management=True`, treatment 1 is rendered with
    `IRRIG=L` and `FERTI=L`.
- `src/run_all_year_direct_action_safe_ppo.py`
  - `make_base_env(...)` now calls `build_env_args(..., linked_management=True)`
    for dynamic RL environments.

Audit and smoke scripts:

- `src/audit_ppo_external_action_linkage_rootcause_034_02.py`
- `src/smoke_dynamic_rl_linked_management_fix_034_03.py`
- `src/run_linked_free_timing_ppo_dqn_train_smoke_034_04.py`
- `src/run_fqa2014_linked_free_timing_ppo_dqn_50k_comparison_034_05.py`
- `src/finalize_fqa2014_034_05_after_dqn_timeout.py`

Prompt and record files:

- `prompts/034_02_ppo_external_action_linkage_rootcause_audit.md`
- `docs/034_02_ppo_external_action_linkage_rootcause_audit_record.md`
- `prompts/034_03_dynamic_rl_linked_management_fix_smoke.md`
- `docs/034_03_dynamic_rl_linked_management_fix_smoke_record.md`
- `prompts/034_04_linked_free_timing_ppo_dqn_train_smoke.md`
- `docs/034_04_linked_free_timing_ppo_dqn_train_smoke_record.md`
- `prompts/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison.md`
- `docs/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison_record.md`

Compact result tables:

- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/`
- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/`
- `benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/evaluation/`
- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/`

Large artifacts intentionally not committed:

- model `.zip` files,
- tensorboard event files,
- full DSSAT snapshot folders,
- runtime folders.

## 034_02 root-cause audit

Task: force one external action on FQA2014 and compare management modes.

Forced action:

- DAP1 irrigation = 45 mm
- DAP1 nitrogen = 80 kg/ha

Result:

- `external_rr` rendered `IRRIG=R`, `FERTI=R`.
  - Python safe action requested I45/N80.
  - DSSAT `Summary.OUT` executed I0/N0.
  - Yield: 7953.630981 kg/ha.
- `external_ll` rendered `IRRIG=L`, `FERTI=L`.
  - Python safe action requested I45/N80.
  - DSSAT `Summary.OUT` executed I45/N80.
  - Yield: 8295.548096 kg/ha.

Conclusion:

`IRRIG=R`, `FERTI=R` blocks external Python actions; `IRRIG=L`, `FERTI=L`
enables them.

Main output:

- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_forced_action_summary.csv`
- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_mode_parse.csv`

## 034_03 linked-management smoke

Task: after patching the dynamic RL environment, replay one non-zero action
case per station and verify that DSSAT executed totals match Python safe totals.

Result:

- pass_count = 5
- total = 5
- failures = 0

Station-level checks:

- FQA: safe I45/N80 == Summary I45/N80
- HLA: safe I150/N200 == Summary I150/N200
- LCA: safe I135/N200 == Summary I135/N200
- SYA: safe I150/N240 == Summary I150/N240
- YCA: safe I150/N240 == Summary I150/N240

Conclusion:

The linked-management fix works across the five templates tested.

Main output:

- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_summary.csv`
- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_comparison.csv`

## 034_04 linked free-timing PPO/DQN 5K smoke

Task: run a small 5K training smoke on FQA2014 for both MaskablePPO and DQN
after the interface fix.

Result:

- interface_pass = 2/2

MaskablePPO 5K:

- yield = 8032.222900 kg/ha
- irrigation = 150 mm
- nitrogen = 240 kg/ha
- DSSAT Summary matched Python action totals.
- action sequence:
  `DAP1 I0/N120; DAP2 I15/N0; DAP8 I45/N80; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0`

DQN 5K:

- yield = 8060.125122 kg/ha
- irrigation = 150 mm
- nitrogen = 240 kg/ha
- DSSAT Summary matched Python action totals.
- action sequence:
  `DAP1 I15/N40; DAP8 I15/N40; DAP15 I45/N80; DAP22 I15/N40; DAP29 I15/N40; DAP36 I15/N0; DAP43 I30/N0`

Interpretation:

This smoke proves the training/evaluation loop can run with real DSSAT action
execution. It does not prove agronomic success.

Main output:

- `benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/evaluation/034_04_eval_summary.csv`
- `benchmark_results/034_04_linked_free_timing_ppo_dqn_train_smoke/evaluation/034_04_training_summary.csv`

## 034_05 FQA2014 PPO/DQN 50K comparison

Task: test whether longer training improves the linked free-timing result.

Status:

- MaskablePPO 50K completed and was evaluated.
- DQN 50K did not finish within the smoke waiting budget and was stopped to
  conserve compute. DQN has no performance result in this task.

MaskablePPO 50K:

- yield = 8080.126953 kg/ha
- irrigation = 150 mm
- nitrogen = 240 kg/ha
- WP_ET = 2.22 kg/m3
- PFP_N = 33.7 kg/kg
- simple_profit = 7535.926953
- interface pass = true
- action sequence:
  `DAP1 I45/N40; DAP8 I45/N120; DAP15 I15/N0; DAP16 I0/N40; DAP22 I15/N0; DAP23 I0/N40; DAP29 I15/N0; DAP36 I15/N0`

FQA2014 four-scenario comparison:

| scenario | yield kg/ha | irrigation mm | nitrogen kg/ha | WP_ET kg/m3 | PFP_N kg/kg | simple_profit |
|---|---:|---:|---:|---:|---:|---:|
| null | 7953.63 | 0 | 0 | 2.26 | NA | 7953.63 |
| recorded_farmer_template | 8273.15 | 75 | 144 | 2.27 | 57.5 | 7963.13 |
| official_extension_expert | 8318.39 | 23 | 82 | 2.32 | 101.4 | 8163.53 |
| dssat_auto | 7953.63 | 0 | 0 | 2.26 | NA | 7953.63 |
| linked_free_timing_maskableppo_50k | 8080.13 | 150 | 240 | 2.22 | 33.7 | 7535.93 |

Conclusion:

The interface fix works, but current free-timing PPO 50K is not successful on
FQA2014. It uses more water and nitrogen than expert while producing lower yield
and lower efficiency.

## What this means for earlier PPO/DQN results

Earlier RL figures/results generated before the linked-management audit should
be treated carefully. If their rendered treatment used `IRRIG=R`, `FERTI=R`, then
the plotted Python action may not be the same as DSSAT's executed management.

Do not use pre-034 linked-unverified PPO/DQN performance as final evidence unless
that specific run has been rechecked against DSSAT `Summary.OUT`.

The four baseline scenarios are conceptually separate because they are static or
auto-management simulations, not external Python RL actions. Still, any final
comparison should use one unified, currently verified input chain.

## Current recommended next step

Do not expand the current free-timing PPO/DQN configuration to all sites yet.

First, use the linked-management environment and solve one station-year under
verified real-action execution. The current FQA2014 result indicates that reward
and/or constraints still drive high-input policies rather than expert-beating
water/nitrogen-efficient policies.

Potential next tasks:

1. Re-examine the free-timing reward and constraints under linked execution.
2. Add explicit checkpoint guardrails based on yield and resource efficiency.
3. Run a shorter, well-logged DQN comparison separately if DQN is still needed.
4. Only after a single linked station-year succeeds, scale to all years/sites.

## Reproduction commands used

Root directory:

`C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi`

Docker command pattern:

`docker exec -w /workspace nifty_taussig /bin/bash -lc "/opt/gym_dssat_pdi/bin/python <script>"`

Executed scripts:

- `/opt/gym_dssat_pdi/bin/python src/audit_ppo_external_action_linkage_rootcause_034_02.py`
- `/opt/gym_dssat_pdi/bin/python src/smoke_dynamic_rl_linked_management_fix_034_03.py`
- `/opt/gym_dssat_pdi/bin/python src/run_linked_free_timing_ppo_dqn_train_smoke_034_04.py`
- `/opt/gym_dssat_pdi/bin/python src/finalize_fqa2014_034_05_after_dqn_timeout.py`
