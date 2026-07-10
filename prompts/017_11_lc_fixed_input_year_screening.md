# 017_11 LC fixed-input yearly baseline screening

## Objective

Fix LC input consistency in temporary copies and screen LC available treatment years before any DQN training.

## Background

017_10 found that LC2008 PDI/gym initialization failed because the source MZX has inconsistent input definitions:

- `ID_SOIL` in `CNLC0801.MZX`: `LC99001200`
- Actual soil profile in `SOIL.SOL`: `LC990012007`
- `ICDAT` for the only initial-condition level: `08121`
- Treatment 1 `SDATE`: `08150`, which is after `ICDAT`

After temporary fixes, LC2008 can run. However, the source file contains four treatments (2008-2011), and all treatments point to `IC=1`. For yearly screening, the temporary file should provide year-matched IC levels and SDATE values.

## Temporary Fixes

Do not modify the source input package. For each temporary run:

1. Replace all `LC99001200` with `LC990012007`.
2. Clone the initial-condition profile into four levels:
   - treatment 1: `IC=1`, `ICDAT=08121`
   - treatment 2: `IC=2`, `ICDAT=09121`
   - treatment 3: `IC=3`, `ICDAT=10121`
   - treatment 4: `IC=4`, `ICDAT=11121`
3. Update treatment IC pointers to `1/2/3/4`.
4. Update each simulation-control `SDATE` to the matching `YY121`.
5. Update each treatment's planting date row (`PL`) to the matching year while keeping the original day-of-year pattern.
6. Use a year-specific temporary FileX name such as `CNLC0901.MZX`, because PDI/DSSAT may infer/check weather files from the FileX/weather naming context.
7. Pass all `CNLC*.WTH` files as auxiliary files. The multi-treatment FileX may validate weather files from all FILES rows even when a single treatment is selected.
8. Align reported irrigation and fertilization event dates in `@I`/`@F` rows to the treatment year while preserving DSSAT fixed-column formatting.
9. For `dssat_auto`, set the selected treatment's MI/MF pointers to `0/0` after switching management to automatic mode, so automatic management does not still validate stale reported event rows.

## Scenarios

For each available LC treatment year:

- `null`
- `recorded`
- `dssat_auto`

## Outputs

Save under:

`DSSAT_auto_validation/lc_fixed_input_year_screening_017_11`

Required outputs:

- status CSV
- daily CSV
- events CSV
- summary CSV
- Chinese experiment record MD

## Decision Rules

A year has optimization space if:

- `recorded` or `dssat_auto` increases GWAD over `null` by at least 100 kg/ha; or
- management meaningfully reduces stress while maintaining yield.

If all LC years show nearly identical yield across scenarios, LC is currently low-priority for DQN training unless initial-condition sensitivity is explicitly approved as a separate diagnostic.
