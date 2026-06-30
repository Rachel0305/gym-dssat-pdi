# Prompt: HLA 2004 new HY0006 cultivar low-cost forward check

Date: 2026-06-28

## Goal

Check whether the newly adjusted HLA HY0006 cultivar parameters remain reasonable for the HLA 2004 IC=1 candidate initial-condition setup before any PPO retraining or full four-scenario rerun.

This is a **low-cost forward simulation only**:

- no PPO training
- no parameter search
- no calibration-effect figure generation
- no multi-year batch run

## New cultivar parameters

Use the HY0006 line currently stored in:

`DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/input_corrected_package/MZCER048.CUL`

Current line:

```text
HY0006 Haiyu    No006       . IB0001 235.2 0.494 625.3 278.0 15.50 40.00
```

## Scenarios

Run HLA 2004 under the same candidate IC setup for three forward-check scenarios:

1. `new_cultivar_null`
   - candidate IC
   - no irrigation
   - no fertilizer

2. `new_cultivar_recorded_expert_replay`
   - candidate IC
   - previously saved recorded/expert management replay input
   - no new management design

3. `new_cultivar_dssat_auto`
   - candidate IC
   - DSSAT native automatic irrigation and automatic fertilizer attempt
   - diagnostic only

## Outputs

Save only the minimum necessary artifacts:

- per-scenario input folder
- per-scenario PDI raw output snapshot
- per-scenario daily post-state CSV
- combined summary CSV
- management-event CSV
- one Markdown record with interpretation

Do not generate calibration scatterplots here. The final five-station calibration plot will be generated later using the user's existing plotting script after all station parameters are finalized.
