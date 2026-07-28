# 032_02 cumulative reward plot metric audit

## Purpose

Audit whether the cumulative reward panel in existing five-scenario daily process figures is safe to show.

The concern is that PPO candidates and baseline scenarios may use different source reward definitions or reward scales, making the cumulative reward panel misleading even when endpoint agronomic metrics are valid.

## Scope

- No training.
- No DSSAT rerun.
- Read existing 031_42 sample five-scenario daily and summary tables.
- Compare source-provided cumulative reward against a recomputed common reward proxy.

## Checks

1. Identify the grain: station-year-scenario daily rows.
2. Inspect whether `reward_step` values are comparable across scenarios.
3. Recompute common final reward proxies from endpoint values:

```text
common_reward_031 = final_grain_kg_ha - 1.1 * total_irrigation_mm - 1.58 * total_nitrogen_kg_ha
common_reward_original = final_grain_kg_ha - total_irrigation_mm - 5 * total_nitrogen_kg_ha
```

4. Compare source final cumulative reward vs recomputed common reward.
5. Flag scenarios where source reward scale is incompatible.

## Decision rule

- If source cumulative reward differs across scenarios by obvious scale/definition artifacts, do not show it as "common reward".
- Replace the cumulative reward panel with one of:
  - cumulative water/N cost panel;
  - cumulative water and N usage panel;
  - endpoint common reward bar chart using a recomputed formula;
  - omit cumulative reward from daily process figures.

