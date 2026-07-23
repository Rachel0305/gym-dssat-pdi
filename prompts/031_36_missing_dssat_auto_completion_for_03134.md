# 031_36 Missing true DSSAT-auto completion for 031_34/031_35

## Goal

Complete the remaining true `dssat_auto` baseline gaps after 031_35 so that the 031_34 frozen MaskablePPO all-year transfer candidates can be compared against four baseline scenarios wherever possible:

- `null`
- recorded-farmer surrogate from 027_05 transferred template when true yearly recorded farmer is unavailable
- true `dssat_auto`
- `official_extension_expert`

## Scope

- No PPO/DQN training.
- Do not change PPO candidate models or existing 031_34/031_35 outputs.
- Generate only true `dssat_auto` for station-years where 031_35 template-aware comparison remains `baseline_incomplete_3rows`.
- Do not overwrite original MZX/WTH/SOL/CUL inputs.
- Modify only rendered per-run FileX templates under the 031_36 output directory.

## DSSAT-auto definition

For each target station-year:

1. Render the same all-year environment template used by the 031 series.
2. Set treatment 1 management flags to automatic irrigation and automatic fertilization (`A/A`).
3. If the rendered template lacks an `AUTOMATIC MANAGEMENT` section, insert the same generic automatic-management block used in prior 031_28/031_29 baseline completion.
4. Execute the season with zero external RL actions; DSSAT automatic management is responsible for in-season water/nitrogen decisions.

## Execution protocol

1. Smoke: run one known missing case (`HLA 2004`) and verify:
   - one `dssat_auto` summary row is produced;
   - `Summary.OUT` can be parsed;
   - snapshot is saved;
   - comparison table can be rebuilt.
2. Full: if smoke passes, run all remaining missing true-auto station-years.
3. Produce:
   - generated auto summary CSV;
   - generated daily CSV;
   - coverage manifest;
   - unified baseline table with 031_35 + 031_36;
   - template-aware four-baseline PPO comparison table;
   - Markdown experiment record.

## Interpretation boundary

`recorded_farmer_template_02705` remains a transferred recorded-management surrogate, not true yearly recorded farmer management. 031_36 only repairs missing true `dssat_auto`; it does not solve the absence of true recorded farmer data for all years.
