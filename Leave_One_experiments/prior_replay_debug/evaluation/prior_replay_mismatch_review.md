# Prior replay mismatch review

Generated at: 2026-06-06

Root cause summary:

1. 006_09 BC_two_stage evaluation used exported action tables from the learned two-stage model and a custom evaluator that applies action safety with `sim_day`.
2. 006_10 FT0 replay used `SafeActionWrapper`, which reads the environment `dap` before each step. In these DSSAT runs the first several rows have `dap = 0`, so day-1 fertilizer events were rejected by `anfer_dap_range`.
3. 006_10 also changed `daily_n_max` to 80 kg/ha. The BC_two_stage action table contains events above 80 kg/ha: HLA 132.75, SYA 138.50, LCA 93.75 and 86.75. These were clipped or removed.
4. Therefore FT0 was not a pure replay of 006_09. It mixed prior replay with a different safety clock and a stricter daily N cap.

Checklist:

| item | 006_09 BC_two_stage | 006_10 FT0 | mismatch |
| --- | --- | --- | --- |
| station/eval years | HLA/SYA/LCA, 10 evals | HLA/SYA/LCA, 10 evals | no |
| prior action source | exported two-stage action table | same action table | no |
| live sklearn model/scaler | trained on Windows, exported to action table for DSSAT container replay | not used | no practical mismatch |
| feature columns | used during action table export | action table replay | no during replay |
| action unit | real kg/ha and mm | real kg/ha and mm before safety | no |
| safety day variable | sim_day | environment dap | yes |
| daily N max | 150 kg/ha in 006_09 replay config | 80 kg/ha | yes |
| residual/guardrail affecting FT0 | none | guardrail/safety still active | yes, through safety cap |

The fix is to use a pure replay evaluator and a sim-day-based safety wrapper for the fixed constrained PPO interface.
