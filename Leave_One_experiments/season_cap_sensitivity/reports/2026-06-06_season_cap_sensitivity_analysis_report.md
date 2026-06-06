# Season cap sensitivity analysis report

Generated at: 2026-06-06

## Goal

This stage diagnoses whether action-safe PPO behavior is dominated by the fixed season cap. It does not modify reward, does not train PPO without action safety, and should not be interpreted as final optimal water-nitrogen management.

## Cap vs budget scenario

- Action safety cap is a technical safety constraint on maximum seasonal water and nitrogen actions.
- Rainfall-scaling budget scenario is a weather or water-availability experiment under altered rainfall conditions.
- These are different experiments. Cap sensitivity should be used before a formal rainfall-scaling budget scenario.

## Design

- Total models planned: 25
- Timesteps per model: 5000
- Cap levels: 100/150, 150/225, 200/300, 250/375, 300/450
- Run order: HLA pilot, then SYA, LCA, then FQA/YCA.

## Pretrain smoke checks

- Passed smoke checks: 50/50

## Quality gate

- Passed evaluations: 70/70
- Completed cap models: 25

## Cap saturation

- All evaluations at or near their cap: True
| station | cap | irrigation_cap | n_cap | irrigation_ratio | n_ratio | trigger_days |
|---|---|---:|---:|---:|---:|---:|
| FQA | cap_current | 200.00 | 300.00 | 1.000 | 1.000 | 102.50 |
| FQA | cap_high | 300.00 | 450.00 | 1.000 | 1.000 | 102.50 |
| FQA | cap_low | 100.00 | 150.00 | 1.000 | 1.000 | 102.50 |
| FQA | cap_mid_high | 250.00 | 375.00 | 1.000 | 1.000 | 102.50 |
| FQA | cap_mid_low | 150.00 | 225.00 | 1.000 | 1.000 | 102.50 |
| HLA | cap_current | 200.00 | 300.00 | 1.000 | 1.000 | 159.00 |
| HLA | cap_high | 300.00 | 450.00 | 1.000 | 1.000 | 159.00 |
| HLA | cap_low | 100.00 | 150.00 | 1.000 | 1.000 | 159.00 |
| HLA | cap_mid_high | 250.00 | 375.00 | 1.000 | 1.000 | 159.00 |
| HLA | cap_mid_low | 150.00 | 225.00 | 1.000 | 1.000 | 159.00 |
| LCA | cap_current | 200.00 | 300.00 | 1.000 | 1.000 | 100.25 |
| LCA | cap_high | 300.00 | 450.00 | 1.000 | 1.000 | 100.25 |
| LCA | cap_low | 100.00 | 150.00 | 1.000 | 1.000 | 100.25 |
| LCA | cap_mid_high | 250.00 | 375.00 | 1.000 | 1.000 | 100.25 |
| LCA | cap_mid_low | 150.00 | 225.00 | 1.000 | 1.000 | 100.25 |
| SYA | cap_current | 200.00 | 300.00 | 1.000 | 1.000 | 140.67 |
| SYA | cap_high | 300.00 | 450.00 | 1.000 | 1.000 | 140.67 |
| SYA | cap_low | 100.00 | 150.00 | 1.000 | 1.000 | 140.67 |
| SYA | cap_mid_high | 250.00 | 375.00 | 1.000 | 1.000 | 140.67 |
| SYA | cap_mid_low | 150.00 | 225.00 | 1.000 | 1.000 | 140.67 |
| YCA | cap_current | 200.00 | 300.00 | 1.000 | 1.000 | 110.00 |
| YCA | cap_high | 300.00 | 450.00 | 1.000 | 1.000 | 110.00 |
| YCA | cap_low | 100.00 | 150.00 | 1.000 | 1.000 | 110.00 |
| YCA | cap_mid_high | 250.00 | 375.00 | 1.000 | 1.000 | 110.00 |
| YCA | cap_mid_low | 150.00 | 225.00 | 1.000 | 1.000 | 110.00 |

## Marginal response

| station | cap | yield | reward | irrigation | n | delta_yield | yield_gain_per_100mm | limited |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FQA | cap_low | 6159.09 | 97.97 | 100.00 | 150.00 | nan | nan | True |
| FQA | cap_mid_low | 6438.64 | 97.80 | 150.00 | 225.00 | 279.55 | 559.10 | True |
| FQA | cap_current | 6449.15 | 90.10 | 200.00 | 300.00 | 10.51 | 21.01 | True |
| FQA | cap_mid_high | 6617.62 | 84.06 | 250.00 | 375.00 | 168.48 | 336.95 | True |
| FQA | cap_high | 6623.52 | 76.56 | 300.00 | 450.00 | 5.90 | 11.80 | True |
| HLA | cap_low | 5440.73 | 74.04 | 100.00 | 150.00 | nan | nan | False |
| HLA | cap_mid_low | 6881.57 | 86.72 | 150.00 | 225.00 | 1440.85 | 2881.69 | False |
| HLA | cap_current | 7122.66 | 84.60 | 200.00 | 300.00 | 241.09 | 482.18 | False |
| HLA | cap_mid_high | 6996.84 | 76.84 | 250.00 | 375.00 | -125.82 | -251.64 | False |
| HLA | cap_high | 6843.57 | 66.84 | 300.00 | 450.00 | -153.27 | -306.54 | False |
| LCA | cap_low | 8539.97 | 150.91 | 100.00 | 150.00 | nan | nan | False |
| LCA | cap_mid_low | 8741.57 | 144.38 | 150.00 | 225.00 | 201.60 | 403.19 | False |
| LCA | cap_current | 8790.05 | 136.05 | 200.00 | 300.00 | 48.48 | 96.97 | False |
| LCA | cap_mid_high | 8791.02 | 128.08 | 250.00 | 375.00 | 0.97 | 1.95 | False |
| LCA | cap_high | 8792.16 | 120.28 | 300.00 | 450.00 | 1.14 | 2.27 | False |
| SYA | cap_low | 8659.74 | 104.58 | 100.00 | 150.00 | nan | nan | False |
| SYA | cap_mid_low | 10176.65 | 108.64 | 150.00 | 225.00 | 1516.91 | 3033.82 | False |
| SYA | cap_current | 10252.54 | 102.55 | 200.00 | 300.00 | 75.89 | 151.77 | False |
| SYA | cap_mid_high | 9865.85 | 88.90 | 250.00 | 375.00 | -386.69 | -773.38 | False |
| SYA | cap_high | 9648.39 | 78.73 | 300.00 | 450.00 | -217.47 | -434.93 | False |
| YCA | cap_low | 8054.17 | 149.19 | 100.00 | 150.00 | nan | nan | True |
| YCA | cap_mid_low | 8669.90 | 155.05 | 150.00 | 225.00 | 615.73 | 1231.46 | True |
| YCA | cap_current | 8748.50 | 148.76 | 200.00 | 300.00 | 78.60 | 157.20 | True |
| YCA | cap_mid_high | 8748.58 | 141.16 | 250.00 | 375.00 | 0.09 | 0.17 | True |
| YCA | cap_high | 8749.44 | 134.31 | 300.00 | 450.00 | 0.85 | 1.71 | True |

## Interpretation

PPO filled every tested season cap. This means the current reward still encourages using all water and nitrogen allowed by action safety.
The 200/300 cap should be treated as a diagnostic safety setting, not as final management guidance.
If yield increases flatten while actions remain saturated, 200/300 may be a reasonable conservative experimental cap; if yield/reward continue rising strongly up to 300/450, reward cost terms need attention before multi-seed validation.

## Next recommendation

Recommended next step: revise reward cost terms or run a constrained rainfall-scaling budget scenario only after deciding whether the cap should be fixed at a conservative value. Multi-seed stability should come after the cap/cost design is less dominated by the safety ceiling.