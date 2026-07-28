# 031_41 LC2010 free-timing MaskablePPO operation-event-cost smoke

## Background

The frozen free-timing MaskablePPO candidate selected for LCA/LC2010 uses repeated small irrigation events. This is simulator-feasible, but management-unrealistic because each irrigation/fertilization operation has a fixed labor/equipment cost that is not represented by the current reward.

Current frozen framework reward:

```text
reward = (0.158 * final_grnwt - 1.1 * irrigation_mm - 1.58 * nitrogen_kg) * 0.001
```

This reward penalizes total water and nitrogen amounts, but does not penalize operation count. Therefore, splitting the same total water into many small irrigations is nearly cost-equivalent to fewer larger irrigations.

## Objective

Run one LC2010 smoke test that changes only the reward accounting by adding fixed per-operation costs:

```text
unscaled_reward =
    0.158 * final_grnwt_at_harvest
    - 1.1 * irrigation_mm
    - 1.58 * nitrogen_kg
    - 10.0 * 1[irrigation_mm > 0]
    - 20.0 * 1[nitrogen_kg > 0]

reward = unscaled_reward * 0.001
```

Interpretation of fixed costs:

- irrigation event cost 10.0 is approximately equivalent to 9.1 mm water cost under `water_cost=1.1`;
- fertilization event cost 20.0 is approximately equivalent to 12.7 kg nitrogen cost under `nitrogen_cost=1.58`;
- these values are fixed before training and are not tuned after seeing results.

## Scope

- Station-year: LCA/LC2010 only.
- Algorithm: MaskablePPO.
- Seed: 1 only for the first smoke, because the current frozen LC2010 representative candidate is seed1 checkpoint75000.
- Training budget: 50,000 timesteps.
- Checkpoints: 10,000, 30,000, 50,000.
- Action levels unchanged: irrigation `[0, 6, 12, 18, 24]`, nitrogen `[0, 40, 80, 120, 160]`.
- Action safety unchanged:
  - season irrigation soft limit 160 mm;
  - season nitrogen soft limit 250 kg/ha;
  - min days between irrigation/fertilization 7;
  - irrigation DAP 1-120;
  - fertilization DAP 1-90.
- No expert DAP windows.
- No DSSAT/weather/baseline data changes.
- No parameter scan.

## Pre-registered checks

### Engineering checks

1. Config, script, selected scenario, daily outputs, checkpoint summaries, and record must be saved.
2. Reward closure must be checked:

```text
literature_reward_unscaled
= literature_yield_component
  - literature_resource_cost
  - operation_event_cost
```

within floating-point tolerance.

### Scientific checks

Compare the selected 031_41 LC2010 candidate against the frozen 031_39 LC2010 representative:

1. irrigation event count should decrease or at least not increase;
2. total irrigation should not increase materially;
3. final yield should not collapse;
4. at least one of yield, WP_ET, or PFP_N should still beat the four-scenario envelope if available;
5. action sequence should be inspected for repeated 6 mm micro-irrigation.

## Stop rule

This smoke only decides whether fixed operation cost is a plausible next optimization variable. It does not authorize all-site reruns. If LC2010 still shows repeated small irrigations or loses the already observed metric advantage, stop and record the negative result instead of adding a second reward change.

