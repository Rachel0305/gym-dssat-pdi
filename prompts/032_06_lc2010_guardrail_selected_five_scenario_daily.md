# 032_06 LC2010 guardrail-selected PPO five-scenario daily figures

## Purpose

Visualize whether the 032_05 guardrail-selected LC2010 free-timing PPO checkpoints are agronomically interpretable against the four baseline scenarios.

This task does not train. It does not rerun DSSAT. It reuses:

- baseline daily evidence from 031_39 LC2010 five-scenario package;
- PPO daily CSVs already generated in 032_04;
- guardrail checkpoint selections from 032_05.

## Candidate selection

Use the 032_05 guardrail pair:

```text
yield_guardrail = 95pct_expert
nstress_guardrail = moderate_nstress
```

This pair selected 3/3 seeds and enforces:

- yield >= 0.95 * expert yield
- max_NSTRES <= 0.15

## Plotting policy

- Generate one five-scenario daily-process figure per selected seed.
- Keep the old 027_05/031_39 multi-panel visual style as much as possible.
- Do not reuse the old mixed-scale cumulative reward panel.
- Replace the final panel with cumulative irrigation and cumulative nitrogen use.
- Save both PNG and SVG.

## Required outputs

- Combined daily CSV for each selected seed.
- Summary CSV for each selected seed.
- Evidence checks CSV.
- One PNG and one SVG daily-process figure per selected seed.
- Markdown record in `docs/`.

## Interpretation boundaries

- This is a visual/diagnostic package, not new training evidence.
- If a seed's selected candidate uses high water/N or leaves visible stress, record that honestly.
- Do not claim cross-year or cross-station success from LC2010 seed-level figures.
