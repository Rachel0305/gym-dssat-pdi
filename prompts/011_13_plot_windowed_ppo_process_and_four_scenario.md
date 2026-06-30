# 011_13 Plot windowed PPO process and four-scenario comparison

## Purpose

Visualize the `joint_windowed` PPO results from 011_12 so the management timing can be inspected directly.

Main questions:

1. Did the earliest-operation / growth-stage window prevent DAP2 budget exhaustion?
2. Are HLA2010 seed0 and seed1 stable?
3. Does HLA2015 seed0 show the same management pattern?
4. Can these results be placed beside null, shifted expert, and DSSAT-auto-attempt scenarios for advisor discussion?

## Inputs

Windowed PPO outputs:

- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/windowed_seed0_5000steps`
- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2010/windowed_seed1_5000steps`
- `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/ppo_smoke/2015/windowed_seed0_5000steps`

Existing four-scenario source, if available:

- `DSSAT_auto_validation/HLA_2004/hla_2010_2015_four_scenario_with_ppo`

## Required outputs

Use high-readability process-style figures with shared DAP x-axis:

- rainfall bars
- water stress line
- nitrogen stress line
- irrigation/fertilization management events
- grain and aboveground biomass

For windowed PPO:

- compare 2010 seed0, 2010 seed1, and 2015 seed0

For four-scenario:

- include null, shifted expert, DSSAT auto attempt, and windowed PPO where source data exist
- clearly label DSSAT auto as auto-irrigation plus auto-N attempt, not successful auto-N

Also export chart-ready daily/event CSV files.

## Constraints

- Do not retrain.
- Do not modify original input files.
- Keep styling simple and readable.
- Save script, figures, and CSV outputs.
