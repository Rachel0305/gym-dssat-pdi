# Prompt: HLA candidate optimization-year screening

Date: 2026-06-28

## Goal

Screen HLA candidate years before formal PPO water-nitrogen optimization.

The aim is to find a year that is not as extreme as 2004, but still has enough water/nitrogen stress and management response to make PPO optimization meaningful.

## Candidate years

Initial candidates from existing IC=1 yearly diagnostics:

- 2007
- 2010
- 2015

These years were selected because existing null-vs-auto summaries suggested nonzero irrigation response without the complete null failure seen in 2004.

## Constraints

- No PPO training.
- No parameter search.
- Use the newly adjusted HLA HY0006 cultivar line.
- Keep IC=1.
- Do not invent recorded/expert management. Use recorded management only if an actual MZX/field record exists in the project.

## Scenarios

For each candidate year, run or summarize:

1. `null`
   - no irrigation
   - no fertilizer

2. `recorded`
   - only if actual recorded management input exists
   - otherwise mark as unavailable, do not fabricate

3. `dssat_auto`
   - DSSAT native automatic irrigation and automatic fertilizer attempt
   - diagnostic only

## Outputs

For each year:

- process figure: rainfall + irrigation + fertilizer + WSPD + NSTD
- yield figure: GWAD and CWAD
- daily CSV
- corrected summary CSV
- management-event CSV

Final screening table should answer:

- Does null complete a normal season?
- Is there meaningful water or nitrogen stress?
- Does management improve yield?
- Is the year a better PPO optimization candidate than 2004?
