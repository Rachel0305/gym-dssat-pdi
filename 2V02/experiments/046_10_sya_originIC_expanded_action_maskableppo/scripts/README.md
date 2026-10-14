# 046_10 figure package

This folder is the experiment-local entry point for the 046_10 figures.

It reuses the existing audited scripts:

- `src/build_sya_configured_five_scenario_daily_plots_046_06.py`
- `src/build_sya_configured_reporting_046_05.py`

Pinned experiment inputs:

- config: `configs/046_10_sya_originIC_expanded_action_maskableppo.json`
- PPO run: `benchmark_results/046_10_sya_originIC_expanded_action_maskableppo`
- checkpoint: `100000`
- validation years: `2014-2023`
- default auto source: `configs/046_11_sya_originIC_external_auto_n_rule_nstd050_minimal.json`

Run from the container/project shell:

```bash
cd /workspace/src
python ../experiments/046_10_sya_originIC_expanded_action_maskableppo/scripts/run_046_10_figures.py --dry-run
python ../experiments/046_10_sya_originIC_expanded_action_maskableppo/scripts/run_046_10_figures.py
```

Each run creates a generated reporting config under:

```text
experiments/046_10_sya_originIC_expanded_action_maskableppo/runs/<label>/configs/
```

The generated config appends the label to the output task name, so new figures
and tables do not overwrite older figure packages. Prefer semantic labels such
as `auto_minimal_nstd050` over raw timestamps.

To use a different auto threshold configuration:

```bash
python ../experiments/046_10_sya_originIC_expanded_action_maskableppo/scripts/run_046_10_figures.py \
  --auto-config configs/<your_auto_threshold_config>.json
```

If the auto source was generated with `--run-id <stamp>`, pass the same stamp:

```bash
python ../experiments/046_10_sya_originIC_expanded_action_maskableppo/scripts/run_046_10_figures.py \
  --auto-config configs/046_11_sya_originIC_external_auto_n_rule_nstd050_minimal.json \
  --auto-run-id <stamp> \
  --label auto_minimal_nstd050
```

Expected output roots:

- yearly five-scenario process figures:
  `benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000/`
- yield, WP_ET, PFP_N, irrigation, and nitrogen bar charts:
  `benchmark_results/046_05_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_reporting_ckpt100000/`
