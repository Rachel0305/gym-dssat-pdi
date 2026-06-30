# Prompt: HLA 2007/2009 management-response space check

Date: 2026-06-28

## Goal

Before restarting formal PPO water-nitrogen optimization, run a low-cost forward check on HLA 2007 and 2009 using the newly adjusted HY0006 cultivar parameters.

The purpose is to decide whether 2007/2009 are better candidate optimization years than the extreme-drought 2004 case.

## Rationale

- 2007/2009 are already being used as the HLA cultivar calibration/validation pair.
- Their recorded field management is real experimental management, so it can temporarily serve as a recorded/expert reference.
- 2004 remains useful as an extreme-drought stress test, but it may be too harsh for the first formal optimization experiment.

## Runtime constraints

- No PPO training.
- No parameter search.
- No calibration scatterplot generation.
- Forward simulations only.
- Keep IC=1 year-matched setup from `CNHL0701_corrected_IC123.MZX`.

## Scenarios

For each year:

1. `null`
   - no irrigation
   - no fertilizer

2. `recorded`
   - original recorded management from the corrected MZX treatment
   - use as temporary expert/reference management

3. `dssat_auto`
   - DSSAT native automatic irrigation and automatic fertilizer attempt
   - diagnostic only

## Key questions

For each year:

- Does null show meaningful water or nitrogen stress?
- Does recorded management reduce stress and improve yield?
- Does DSSAT automatic management trigger irrigation or fertilizer?
- Is there room for PPO to improve under a realistic water-nitrogen budget?

## Output

Save:

- per-run input folders
- per-run PDI output snapshots
- per-run daily post-state CSV
- combined daily CSV
- combined summary CSV
- management event CSV
- one Markdown interpretation file

Do not generate final calibration-effect figures here.
