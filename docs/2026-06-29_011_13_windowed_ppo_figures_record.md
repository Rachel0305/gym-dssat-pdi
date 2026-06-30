# 011_13 Windowed PPO process figures record

## Purpose

Generate advisor-facing process-style figures for the 011_12 `joint_windowed` PPO runs.

The visual check focuses on:

- whether the DAP2 early-budget-exhaustion behavior disappeared;
- whether HLA2010 seed0 and seed1 are stable;
- whether HLA2015 seed0 shows the same operation-window behavior;
- how current windowed PPO compares with null, shifted expert, and DSSAT auto-irrigation/auto-N-attempt scenarios.

## Prompt

`prompts/011_13_plot_windowed_ppo_process_and_four_scenario.md`

## Script

`src/plot_hla_windowed_ppo_011_13.py`

This script only reads existing CSV and DSSAT/PDI output snapshots. It does not run DSSAT and does not train PPO.

## Command

```bash
python src/plot_hla_windowed_ppo_011_13.py
```

## Output folder

`DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/figures_011_13_windowed_ppo`

## Figures

- `hla_2010_four_scenario_with_windowed_ppo_process.png`
- `hla_2015_four_scenario_with_windowed_ppo_process.png`
- `hla_2010_windowed_ppo_seed0_seed1_process.png`
- `hla_2015_windowed_ppo_seed0_process.png`

## Data outputs

- `hla_2010_2015_four_scenario_with_windowed_ppo_daily.csv`
- `hla_windowed_ppo_seed_year_daily.csv`
- `hla_windowed_ppo_seed_year_summary.csv`
- `figure_manifest.csv`

## Notes

The source four-scenario CSV encodes the null scenario as the literal string `null`, which pandas can otherwise interpret as missing data. The plotting script explicitly maps this to `null_zero` so null appears correctly in the figures and legends.

The current four-scenario figures use `joint_windowed` 5K seed0 as the PPO scenario. DSSAT auto is labelled as "auto irrigation + auto-N attempt" because previous diagnostics showed native automatic nitrogen did not trigger.

The HLA2010 seed-stability figure shows seed0 and seed1 are nearly identical in stress, management timing, grain yield, and biomass under the operation-window wrapper.
