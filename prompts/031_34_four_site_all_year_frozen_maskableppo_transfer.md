# 031_34 Four-site all-year frozen MaskablePPO transfer

## Purpose

Freeze the candidate checkpoints identified after 031_33/031_34a and evaluate them across all weather-available years within their own station.

This task asks:

> After training one free-timing MaskablePPO model per station training year, can the frozen policy transfer to other years of the same station and beat the four-scenario baseline envelope on yield / WP_ET / PFP_N?

## Scope

- Target stations:
  - HLA
  - FQA
  - LCA
  - YCA
- Source checkpoints:
  - 031_34a 95% yield-guardrailed reselection table.
- Evaluation years:
  - all scenario-pool years from 2000 onward for each station.
- No new training.
- No reward change.
- No parameter tuning.
- No expert-DAP windows.
- Deterministic frozen-policy evaluation only.

## Candidate policy source

Use:

```text
benchmark_results/031_34a_reselect_03133_checkpoints_by_advisor_metrics/evaluation/031_34a_guardrailed_reselection_sensitivity.csv
```

Filter:

```text
yield_guardrail_fraction == 0.95
```

This gives one selected checkpoint per station-seed. For HLA/FQA, these are still marked as not passing the known advisor metrics on the training year; they are included to test whether transfer years contain usable outcomes.

## Metrics to compute

For every station-year-seed:

- final grain yield;
- biomass;
- total irrigation;
- total nitrogen;
- ETCP;
- WP_ET;
- PFP_N;
- water stress days;
- nitrogen stress days;
- action sequence;
- daily CSV path;
- snapshot path.

Unlike 031_33, this task must save snapshots and compute ETCP/WP_ET.

## Baseline comparison

Where four baseline rows are available, compare candidate against:

- null;
- recorded_farmer;
- dssat_auto;
- official_extension_expert.

Compute:

- gap to maximum baseline yield;
- gap to maximum baseline WP_ET;
- gap to maximum baseline PFP_N;
- one-metric strict winner flag;
- winning metric names.

If baseline rows are missing for a year, keep the candidate evaluation and mark comparison as `baseline_missing_not_compared`.

## Execution plan

1. Smoke:
   - run one selected YCA2014 candidate on YCA2014;
   - verify daily CSV, snapshot, ETCP, WP_ET, PFP_N.
2. Full:
   - evaluate all 12 selected station-seed policies across all available years for their own station.

## Stop rules

- Do not retrain if transfer performance is poor.
- Do not change candidate selection inside this task.
- Do not silently drop failed evaluations.
- Save failed rows with traceback.

## Interpretation boundary

031_34 is an intra-station cross-year transfer evaluation. It is not cross-station generalization and not joint training.

