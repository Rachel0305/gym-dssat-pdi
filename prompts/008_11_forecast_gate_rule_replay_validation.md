# 008_11 Forecast Gate Rule Replay Validation

Date: 2026-06-14

## Background

008_09 and 008_10 showed that HLA 2004 can be managed well by a forecast-stress gate with triggered minimum irrigation. However, before treating this as a PPO framework, we must verify that the rule itself does not simply irrigate everywhere.

This stage tests the gate logic without PPO.

## Constraints

- Do not train PPO.
- Do not load PPO model weights.
- Do not modify reward files in site-packages.
- Do not modify `my_data/` original files.
- Do not overwrite previous 008 outputs.
- Use deterministic rule replay only.

## Objective

Run a pure rule-based replay to answer:

```text
Does the forecast gate irrigate only in water-limited or dry-forecast cases,
or does it over-irrigate non-water-limited years?
```

This is a method sanity check, not a final RL result.

## Rule

Stage windows:

| stage | DAP | N | irrigation rule |
|---|---:|---:|---|
| S1 | 0-20 | 50 kg/ha | no irrigation |
| S2 | 21-45 | 50 kg/ha | if SWFAC > 0.05 or future 7d rain < 10 mm, irrigate 30 mm |
| S3 | 46-75 | 50 kg/ha | if SWFAC > 0.05 or future 7d rain < 10 mm, irrigate 30 mm |
| S4 | 76-100 | 0 | if SWFAC > 0.05 or future 7d rain < 20 mm, irrigate 30 mm |
| S5 | 101-end | 0 | if SWFAC > 0.05 or future 7d rain < 20 mm, irrigate 30 mm |

Historical weather is used as perfect forecast.

## Test Cases

Run:

1. `HLA 2004`: strong water-limited showcase. The rule should irrigate and increase yield.
2. `FQA 2008`: previously diagnosed as unsuitable for water showcase. The rule should not over-irrigate without justification.
3. `FQA 2016`: water-stress candidate from 008_05. The rule should behave as an intermediate/response case.

## Outputs

Use:

```text
Leave_One_experiments/forecast_gate_rule_replay_008_11/
```

Generate:

```text
daily_outputs/{station}/{station}_{year}_forecast_gate_rule_daily.csv
event_outputs/{station}_{year}_forecast_gate_rule_events.csv
evaluation/008_11_forecast_gate_rule_replay_summary.csv
evaluation/008_11_forecast_gate_rule_trigger_audit.csv
figures/{station}_{year}_forecast_gate_rule_process.png
docs/2026-06-14_008_11_forecast_gate_rule_replay_validation_report.md
docs/014_008_11_forecast_gate_rule_replay_validation.pptx
```

## Interpretation Rules

- If HLA 2004 irrigates and produces high yield, the gate handles a dry year.
- If FQA 2008 receives little or explainable irrigation, the gate is not blindly irrigating.
- If FQA 2008 receives large irrigation without SWFAC stress or dry forecast, the gate is too aggressive.
- This does not prove PPO contribution. It only checks that the rule boundary is agronomically reasonable.
