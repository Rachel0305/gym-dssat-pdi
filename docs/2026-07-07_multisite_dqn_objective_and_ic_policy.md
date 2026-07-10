# 2026-07-07 multisite DQN objective and IC policy

## Confirmed objective

The advisor-confirmed research target is no longer only "find a feasible DQN policy" or "exceed null".

The target is:

- maximize grain yield;
- improve water and nitrogen use efficiency;
- try to outperform both recorded expert management and DSSAT auto management in yield and resource efficiency.

This means a DQN result should be described cautiously:

- "high-yield success" if DQN has higher yield but uses more water or nitrogen;
- "resource-efficient success" only if DQN has comparable or higher yield with lower or comparable water and nitrogen input;
- "full success candidate" only if DQN improves yield and water/nitrogen efficiency relative to both expert and auto management.

## Initial condition policy

Initial soil water and nitrogen can be diagnosed and adjusted only when there is a defensible input-data reason.

Acceptable reasons:

- the source MZX has inconsistent or invalid IC definitions;
- IC date conflicts with simulation start date;
- soil profile ID in MZX does not match SOIL.SOL;
- a station-year fails to grow or fails to initialize because of input inconsistency;
- calibration evidence shows that the IC profile is not compatible with the observed experiment.

Unacceptable reason:

- changing IC only to make RL outperform the baselines.

For low-headroom stations, IC sensitivity can be run as a separate diagnostic, but it must be labeled as an IC scenario test rather than mixed into the main DQN comparison.

## Current station status

HLA:

- Has successful DQN transfer evidence within station.
- Continue using as the main method-proving site unless later checks contradict it.

YC:

- Has local DQN and cross-year transfer evidence.
- Continue as a second promising site.

FQ:

- Has optimization space in selected years such as 2016 and 2019.
- Year choice is sensitive; continue with careful year screening.

SY:

- Has optimization space.
- Current SY2014 DQN is high-yield but uses full I120/N300 budget, so it is not yet a resource-efficient success.

LC:

- 017_10 found source input issues: soil ID mismatch and IC date before SDATE.
- LC must be repaired and screened before any DQN training.
- 017_11 is the next step: fixed temporary LC inputs and yearly baseline screening for 2008-2011.

## Immediate next step

Run `017_11_lc_fixed_input_year_screening` in the specified Docker/PDI environment when Docker is available.

Only after LC years with real baseline optimization space are found should DQN smoke training be considered.
