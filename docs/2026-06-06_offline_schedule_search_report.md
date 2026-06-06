# Offline schedule search report

Generated at: 2026-06-06

## Goal

This stage pauses PPO tuning and uses deterministic DSSAT schedule search to identify an expert prior for water and nitrogen management. It does not train PPO, does not enter rainfall scaling, and does not modify my_data or previous 006_03-006_07 outputs.

## Search Space

- Station/year: HLA 2011 coarse grid.
- Nitrogen events: DAP 1, 30, 60.
- Irrigation events: DAP 25, 50, 75.
- N event amounts: [0, 75, 150].
- Irrigation event amounts: [0, 50, 100].
- Conservative caps: total N <= 300.0 kg/ha, total irrigation <= 250.0 mm.
- Profit score: 0.01 * grain yield - 0.5 * irrigation - 0.25 * nitrogen.

## Why PPO Is Paused

006_03-006_07 repeatedly saturated 300/450 under cost rewards, terminal profit, low-frequency/window action designs, and explicit budget/event wrappers. The offline search tests whether reasonable non-RL schedules exist before returning to imitation learning or constrained PPO.

## HLA Coarse Grid Result

- Candidate schedules generated: 598
- Completed HLA 2011 evaluations: 598
- Pareto schedules: 8
- Expert candidates selected for cross-year validation: 15
- Low-input candidates within 10% of PPO baseline yield: 14

## Top HLA 2011 Schedules

| schedule_id | final_grnwt | total_irrigation | total_n_fertilizer | profit_score | yield_per_100mm_irrigation | yield_per_100kg_n |
| --- | --- | --- | --- | --- | --- | --- |
| HLA2011_S0027 | 5253.0573 | 0.0 | 75.0 | 33.7806 |  | 7004.0763 |
| HLA2011_S0105 | 6414.1431 | 0.0 | 150.0 | 26.6414 |  | 4276.0954 |
| HLA2011_S0157 | 6364.1425 | 0.0 | 150.0 | 26.1414 |  | 4242.7616 |
| HLA2011_S0261 | 6310.7275 | 0.0 | 150.0 | 25.6073 |  | 4207.1517 |
| HLA2011_S0313 | 6279.7827 | 0.0 | 150.0 | 25.2978 |  | 4186.5218 |
| HLA2011_S0443 | 6179.1998 | 0.0 | 150.0 | 24.292 |  | 4119.4666 |
| HLA2011_S0053 | 5620.0598 | 0.0 | 150.0 | 18.7006 |  | 3746.7065 |
| HLA2011_S0183 | 6853.3051 | 0.0 | 225.0 | 12.2831 |  | 3045.9134 |
| HLA2011_S0131 | 6852.8082 | 0.0 | 225.0 | 12.2781 |  | 3045.6925 |
| HLA2011_S0391 | 6848.5083 | 0.0 | 225.0 | 12.2351 |  | 3043.7815 |

## Cross-Year Ranking

| schedule_id | mean_yield | std_yield | mean_profit | mean_irrigation | mean_n | mean_yield_loss_vs_ppo_baseline | overall_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HLA2011_S0157 | 6752.7806 | 367.4822 | 30.0278 | 0.0 | 150.0 | 0.0179 | 0.7639 |
| HLA2011_S0105 | 6862.4835 | 485.1353 | 31.1248 | 0.0 | 150.0 | 0.0019 | 0.7457 |
| HLA2011_S0183 | 7305.696 | 460.7526 | 16.807 | 0.0 | 225.0 | -0.0625 | 0.7333 |
| HLA2011_S0131 | 7305.5294 | 460.9965 | 16.8053 | 0.0 | 225.0 | -0.0625 | 0.7332 |
| HLA2011_S0261 | 6631.7236 | 530.92 | 28.8172 | 0.0 | 150.0 | 0.0355 | 0.6753 |
| HLA2011_S0313 | 6531.4821 | 484.0511 | 27.8148 | 0.0 | 150.0 | 0.0501 | 0.6694 |
| HLA2011_S0209 | 7305.696 | 460.7526 | -1.943 | 0.0 | 300.0 | -0.0625 | 0.6091 |
| HLA2011_S0573 | 7304.2938 | 462.8249 | -1.9571 | 0.0 | 300.0 | -0.0623 | 0.608 |
| HLA2011_S0417 | 7304.1166 | 463.0908 | -1.9588 | 0.0 | 300.0 | -0.0623 | 0.6078 |
| HLA2011_S0027 | 5124.0224 | 125.9758 | 32.4902 | 0.0 | 75.0 | 0.2548 | 0.6 |

## Best Schedule

| schedule_id | mean_yield | std_yield | mean_profit | mean_irrigation | mean_n | mean_yield_loss_vs_ppo_baseline | overall_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HLA2011_S0157 | 6752.7806 | 367.4822 | 30.0278 | 0.0 | 150.0 | 0.0179 | 0.7639 |

## Extension Decision

- Extend SYA/LCA: True
- Reason: HLA best schedule HLA2011_S0157 passed the low-input <=10% yield-loss gate.
- Imitation dataset: Leave_One_experiments\offline_schedule_search\expert_policy\imitation_dataset.csv

## SYA/LCA Extension

| station | run_status | best_schedule_id | mean_yield | mean_irrigation | mean_n | overall_score |
| --- | --- | --- | --- | --- | --- | --- |
| SYA | ok | SYA2012_S0443 | 8988.5219 | 0.0 | 150.0 | 0.8389 |
| LCA | ok | LCA2010_S0313 | 8757.0909 | 0.0 | 150.0 | 0.7871 |

## Interpretation

If HLA identifies schedules with lower water/N and acceptable yield loss, these schedules should be treated as expert priors, not final RL policies. They can be used for imitation learning, constrained PPO target shaping, or reward calibration. Rainfall-scaling budget scenarios should remain a separate stress-test path.
