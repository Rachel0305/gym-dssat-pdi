# 049 HLA originIC site-transfer record

Date: 2026-08-06

## Decision

SYA is temporarily paused after the 046-048 controlled experiments. The next station is HLA.

049 is registered as a site-transfer workflow: reuse the current SYA PPO method and baseline conventions, and change only the station/input data. Do not tune recorded, expert, auto, reward, observation, or action safety to make PPO look better.

## Frozen Method

- PPO source method: SYA 046_10 expanded-action MaskablePPO.
- Observation: raw 046_02 observation.
- No weather forecast features.
- No observation normalization.
- Action grid: irrigation `[0, 15, 30, 45]` mm; nitrogen `[0, 40, 80, 120]` kg/ha.
- Reward and safety: inherited from the 042_15/046_02 chain.
- Seed: 0.
- Smoke before formal: required.

## HLA Scope

- Station code: `HLA`.
- Site code: `HLA`.
- Input profile: `originIC`.
- Input root: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`.
- Source template expected by the renderer: `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/HL/CNHL0701_corrected_IC123.MZX`.
- Train years: 2004-2013.
- Validation years: 2014-2023.

## Baselines

- `null`: no static water or nitrogen schedule.
- `recorded_farmer_template`: use the existing recorded template pipeline from 027_05/034_00; do not edit the source template.
- `official_extension_expert`: use the existing regional expert schedule helper.
- `dssat_auto_irrigation_external_n_rule`: DSSAT native auto-irrigation plus minimal external auto-N.

Default auto-N rule:

- threshold: `NSTRES >= 0.5`
- dose: `25 kg/ha`
- removed constraints: no DAP cutoff, no minimum interval, no seasonal N cap

## Files

- Prompt: `prompts/049_hla_originIC_site_transfer_expanded_action_ppo_and_auto.md`
- PPO config: `configs/049_00_hla_originIC_expanded_action_maskableppo.json`
- Auto config: `configs/049_01_hla_originIC_external_auto_n_rule_nstd050_minimal.json`
- Code folder: `src/049_hla_originIC_site_transfer/`
- Figure entry point: `src/049_hla_originIC_site_transfer/run_049_02_hla_originIC_five_scenario_figures.py`

## Run Commands

Run from `/workspace/src` in the container:

```bash
python 049_hla_originIC_site_transfer/run_049_00_hla_originIC_expanded_action_maskableppo.py --dry-run
python 049_hla_originIC_site_transfer/run_049_00_hla_originIC_expanded_action_maskableppo.py --smoke
python 049_hla_originIC_site_transfer/run_049_00_hla_originIC_expanded_action_maskableppo.py --formal
python 049_hla_originIC_site_transfer/run_049_01_hla_originIC_external_auto_n_rule_minimal.py --dry-run
python 049_hla_originIC_site_transfer/run_049_01_hla_originIC_external_auto_n_rule_minimal.py
python 049_hla_originIC_site_transfer/run_049_02_hla_originIC_five_scenario_figures.py --dry-run
python 049_hla_originIC_site_transfer/run_049_02_hla_originIC_five_scenario_figures.py
```

For repeated auto-threshold tests, copy the auto config to a new config name and set `external_auto_n_rule.output_suffix`; then pass a stable `--run-id`.

## Figure Outputs

The 049_02 script produces:

- yearly five-scenario daily figures: `049_02_hlaYYYY_five_scenario_daily.png`
- combined daily table: `049_02_hla_five_scenario_daily.csv`
- season summary table: `049_02_hla_five_scenario_season_summary.csv`
- management-event table: `049_02_hla_five_scenario_management_events.csv`
- metric bars: `049_02_hla_five_scenario_metrics.png`
- management bars: `049_02_hla_five_scenario_management.png`
- PPO gap table: `049_02_hla_metric_gaps_vs_four_baseline_max.csv`
