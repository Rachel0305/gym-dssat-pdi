# Why not constrained PPO multi-seed yet

Generated at: 2026-06-06

006_11 fixed the replay inconsistency between 006_09 BC two-stage and 006_10 FT0. The mismatch was caused by two implementation differences: 006_10 used environment `dap`, which was 0 on early rows and blocked first-day N events, and it also clipped daily N at 80 instead of the 150 used by the BC prior action table.

After the fix, pure replay reproduced 006_09 exactly: mean yield 8091.8537, mean irrigation 0.0, and mean N 153.575. This means the BC prior is reliable.

However, the fixed constrained PPO tests still did not produce a better learned policy. FT2 reduced or altered inputs but lost too much yield/profit on HLA, while FT3 avoided 300/450 saturation but mostly hit the stricter 100/200 guardrail and had much lower profit than BC replay. Running multi-seed now would mainly measure the variance of a weak fine-tuning setup.

The bottleneck is the expert dataset: 006_09 used one best schedule per station, producing sparse and narrow action labels. The next step should therefore augment expert schedules across top-profit, high-yield, Pareto-balanced, low-input, and medium-input categories before reattempting constrained PPO.
