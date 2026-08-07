# 049 HLA originIC site-transfer PPO and baseline workflow

## Purpose

Move the controlled SYA originIC workflow to HLA without changing the PPO method.
This task is a site-transfer experiment, not a new PPO-method experiment.

## Fixed Method Contract

- Station: HLA / site HLA.
- Input profile: originIC, using `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`.
- PPO framework: inherit the current SYA 046_10 framework and parameters.
- Observation: raw 046_02 observation, no normalization, no weather forecast.
- Action grid:
  - irrigation: `[0, 15, 30, 45]` mm
  - nitrogen: `[0, 40, 80, 120]` kg/ha
- Reward and safety: inherit the 042_15 / 046_02 chain unchanged.
- Training:
  - smoke: 2K, checkpoints 1K/2K
  - formal: 100K, checkpoints 25K/50K/75K/100K
- Formal training is allowed only after the matching 2K smoke gate passes.

## HLA Year Split

Use the existing half-split registry. Do not invent new years.

- train: 2004-2013
- validation/reporting: 2014-2023

## Baselines

Use already established data/provenance:

- `null`: no irrigation and no fertilizer management.
- `recorded_farmer_template`: static recorded template loaded from the existing 027_05 daily results through the 034_00 baseline helper.
- `official_extension_expert`: existing regional expert schedule from the extension baseline helper.
- `dssat_auto_irrigation_external_n_rule`: DSSAT native automatic irrigation plus minimal external auto-N because DSSAT automatic fertilizer is not operational in this workflow.

Auto-N rule for the default HLA transfer:

- threshold: NSTRES >= 0.5
- dose: 25 kg/ha
- removed constraints: no DAP cutoff, no minimum interval, no seasonal N cap

## Output Rules

- New outputs must live under 049-named folders.
- Do not overwrite historical 046/048/SYA files.
- Write manifests, copied configs, action audits, summaries, and docs.
- Use `--run-id` for repeated auto-threshold tuning runs.
- Keep smoke and formal PPO outputs in separate folders.

## Execution Order

1. Dry-run PPO preflight.
2. Run PPO 2K smoke.
3. If the smoke gate passes, run PPO 100K formal.
4. Run HLA minimal external auto-N baseline.
5. Reuse the five-scenario reporting/figure layer only after PPO and auto outputs are present.

