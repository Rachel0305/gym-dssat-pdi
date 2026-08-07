# 050 YCA/YC originIC site-transfer workflow

This folder keeps the YCA/YC transfer entry points together. 050 differs from
049 in one important way: it regenerates the four non-PPO baselines with the
037_07 static level-1 management-row correction before any five-scenario plot is
trusted.

Run from `/workspace/src` in the container:

```bash
python 050_yca_originIC_site_transfer/run_050_02_yca_originIC_four_baselines_static_level1.py --dry-run
python 050_yca_originIC_site_transfer/run_050_02_yca_originIC_four_baselines_static_level1.py
python 050_yca_originIC_site_transfer/run_050_00_yca_originIC_expanded_action_maskableppo.py --dry-run
python 050_yca_originIC_site_transfer/run_050_00_yca_originIC_expanded_action_maskableppo.py --smoke
python 050_yca_originIC_site_transfer/run_050_00_yca_originIC_expanded_action_maskableppo.py --formal
python 050_yca_originIC_site_transfer/run_050_01_yca_originIC_external_auto_n_rule_minimal.py --dry-run
python 050_yca_originIC_site_transfer/run_050_01_yca_originIC_external_auto_n_rule_minimal.py
python 050_yca_originIC_site_transfer/run_050_03_yca_originIC_five_scenario_figures.py --dry-run
python 050_yca_originIC_site_transfer/run_050_03_yca_originIC_five_scenario_figures.py
```

Use `--run-id <label>` for repeated auto-N threshold tests.

If `benchmark_results/050_02_yca_originIC_four_baselines_static_level1/evaluation/050_02_coverage_manifest.csv`
contains non-ok rows, inspect `050_02_management_event_audit.csv` before using the five-scenario figures.

The figure entry point defaults to:

- PPO root: `benchmark_results/050_00_yca_originIC_expanded_action_maskableppo`
- auto root: `benchmark_results/050_01_yca_originIC_external_auto_n_rule_nstd050_minimal`
- baseline root: `benchmark_results/050_02_yca_originIC_four_baselines_static_level1`
