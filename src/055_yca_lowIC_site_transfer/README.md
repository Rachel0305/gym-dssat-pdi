# 055 YCA/YC lowIC site-transfer workflow

This folder keeps the YCA/YC lowIC transfer entry points together. It follows
the standardized four-step layout:

1. `055_02` rebuilds the four non-PPO baselines with the static level-1
   management correction.
2. `055_00` trains/evaluates the expanded-action MaskablePPO.
3. `055_01` runs DSSAT native auto-irrigation plus minimal external auto-N.
4. `055_03` builds the five-scenario daily figures and seasonal bar charts.

Run from `/workspace/src` in the container:

```bash
python 055_yca_lowIC_site_transfer/run_055_02_yca_lowIC_four_baselines_static_level1.py --dry-run
python 055_yca_lowIC_site_transfer/run_055_02_yca_lowIC_four_baselines_static_level1.py

python 055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py --dry-run
python 055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py --smoke
python 055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py --formal

python 055_yca_lowIC_site_transfer/run_055_01_yca_lowIC_external_auto_n_rule_minimal.py --dry-run
python 055_yca_lowIC_site_transfer/run_055_01_yca_lowIC_external_auto_n_rule_minimal.py

python 055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py --dry-run
python 055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py
```

Use `--run-id <label>` for repeated auto-N threshold tests.

If `benchmark_results/055_02_yca_lowIC_four_baselines_static_level1/evaluation/055_02_coverage_manifest.csv`
contains non-ok rows, inspect `055_02_management_event_audit.csv` before using
the five-scenario figures.

The figure entry point defaults to:

- PPO root: `benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo`
- auto root: `benchmark_results/055_01_yca_lowIC_external_auto_n_rule_nstd050_minimal`
- baseline root: `benchmark_results/055_02_yca_lowIC_four_baselines_static_level1`
