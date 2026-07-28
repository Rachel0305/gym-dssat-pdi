# 032_18 LC missing DSSAT-auto daily completion

## Goal

Complete the missing true `dssat_auto` daily baseline traces needed by the 032_17 LC2005-LC2020 75k PPO daily package.

## Target station-years

Only these three gaps are in scope:

- LCA / LC / 2008
- LCA / LC / 2009
- LCA / LC / 2011

## Boundaries

- No PPO/DQN training.
- No candidate model reselection.
- Do not modify original DSSAT input files.
- Do not overwrite 031_36 historical outputs.
- Generate supplemental `dssat_auto` daily and summary CSVs under a new 032_18 directory.
- Reuse the 031_36 true DSSAT-auto procedure: render the same environment template, set treatment 1 to automatic irrigation/fertilization, run zero external RL actions, and let DSSAT automatic management decide water/nitrogen.

## Validation

- Each target year must produce:
  - a non-empty daily CSV with final `done=True`;
  - one summary row;
  - a copied DSSAT snapshot folder;
  - status `generated_032_18_true_dssat_auto` in the manifest.
- After 032_18 succeeds, rerun 032_17 and require all 16 LC years to be `five_scenario_daily_complete=True`.

## Interpretation

032_18 only repairs missing `dssat_auto` baseline evidence for plotting/comparison. It does not change PPO policy, reward design, training years, checkpoint selection, or any existing 032_11/032_12 candidate outputs.
