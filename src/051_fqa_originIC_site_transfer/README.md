# 051 FQA/FQ originIC site-transfer workflow

This folder keeps the FQA/FQ transfer entry points together. 051 differs from
049 in one important way: it regenerates the four non-PPO baselines with the
037_07 static level-1 management-row correction before any five-scenario plot is
trusted.

Run from `/workspace/src` in the container:

```bash
python 051_fqa_originIC_site_transfer/run_051_02_fqa_originIC_four_baselines_static_level1.py --dry-run
python 051_fqa_originIC_site_transfer/run_051_02_fqa_originIC_four_baselines_static_level1.py
python 051_fqa_originIC_site_transfer/run_051_00_fqa_originIC_expanded_action_maskableppo.py --dry-run
python 051_fqa_originIC_site_transfer/run_051_00_fqa_originIC_expanded_action_maskableppo.py --smoke
python 051_fqa_originIC_site_transfer/run_051_00_fqa_originIC_expanded_action_maskableppo.py --formal
python 051_fqa_originIC_site_transfer/run_051_01_fqa_originIC_external_auto_n_rule_minimal.py --dry-run
python 051_fqa_originIC_site_transfer/run_051_01_fqa_originIC_external_auto_n_rule_minimal.py
python 051_fqa_originIC_site_transfer/run_051_03_fqa_originIC_five_scenario_figures.py --dry-run
python 051_fqa_originIC_site_transfer/run_051_03_fqa_originIC_five_scenario_figures.py
```

Use `--run-id <label>` for repeated auto-N threshold tests.

If `benchmark_results/051_02_fqa_originIC_four_baselines_static_level1/evaluation/051_02_coverage_manifest.csv`
contains non-ok rows, inspect `051_02_management_event_audit.csv` before using the five-scenario figures.

The figure entry point defaults to:

- PPO root: `benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo`
- auto root: `benchmark_results/051_01_fqa_originIC_external_auto_n_rule_nstd050_minimal`
- baseline root: `benchmark_results/051_02_fqa_originIC_four_baselines_static_level1`

