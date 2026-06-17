# 008_12 PPO Contribution Under Forecast Gate

Date: 2026-06-14

## Background

008_11 validated the forecast gate as a deterministic rule boundary:

- HLA 2004 irrigated strongly;
- FQA 2008 did not over-irrigate early;
- FQA 2016 showed intermediate irrigation.

However, this does not prove PPO contribution. The key open question is whether PPO improves beyond the rule baseline, or whether the gate is doing nearly all of the work.

## Constraints

- Do not train unrestricted daily PPO.
- Do not modify reward files in site-packages.
- Do not modify `my_data/` original files.
- Do not overwrite 008_09, 008_10, or 008_11 outputs.
- Keep compute small and serial.

## Objective

Compare:

1. Pure rule replay baseline from 008_11;
2. PPO under the same `rain10_min30` forecast gate.

The main diagnostic question:

```text
Does PPO actively improve water allocation beyond the forecast-gate minimum irrigation,
or is the final performance almost entirely produced by the rule?
```

## Design

Use the same gate as `rain10_min30`:

- S1 blocked;
- S2/S3: future 7d rain threshold = 10 mm;
- S4/S5: future 7d rain threshold = 20 mm;
- triggered minimum irrigation = 30 mm;
- base N = S1/S2/S3 = 50/50/50 kg/ha.

Run PPO on the same three cases:

1. HLA 2004;
2. FQA 2008;
3. FQA 2016.

Use seed 0 and 2000 timesteps per case. Train separately per station-year so this is a contribution diagnostic, not cross-site generalization.

## Required Diagnostics

For each case, save:

```text
irrigation_before_gate
irrigation_after_gate
gate_min_irrigation_applied
ppo_extra_above_min = irrigation_after_gate - gate_min_irrigation_applied
total_ppo_extra_above_min
final_grnwt
total_irrigation
```

Compare these against 008_11 pure rule replay.

## Outputs

Use:

```text
Leave_One_experiments/ppo_contribution_forecast_gate_008_12/
```

Generate:

```text
evaluation/008_12_ppo_contribution_summary.csv
evaluation/008_12_ppo_vs_rule_comparison.csv
daily_outputs/{station}/{station}_{year}_ppo_forecast_gate_daily.csv
daily_outputs/{station}/{station}_{year}_ppo_forecast_gate_steps.csv
figures/{station}_{year}_ppo_vs_rule_process.png
docs/2026-06-14_008_12_ppo_contribution_under_forecast_gate_report.md
docs/015_008_12_ppo_contribution_under_forecast_gate.pptx
```

## Interpretation Rules

- If PPO improves yield or saves water relative to pure rule replay, PPO has a measurable contribution.
- If PPO mostly stays at the minimum irrigation and does not improve yield/water efficiency, report PPO contribution as limited under the current gate.
- If PPO degrades relative to rule replay, the gate may be too deterministic or the PPO training signal may be masked.
