# 052 LCA/LC originIC site-transfer workflow

This folder keeps the LCA/LC transfer entry points together. 052 differs from
049 in one important way: it regenerates the four non-PPO baselines with the
037_07 static level-1 management-row correction before any five-scenario plot is
trusted.

LC has a historical single-year prepared-input adapter noted in
`configs/sites/lca.yaml`, but this 052 screening run intentionally keeps the
same multisite `originIC` pipeline as 050/051. Do not switch input families
inside this series without opening a separate LC-specific experiment.

Run from `/workspace/src` in the container:

```bash
python 052_lca_originIC_site_transfer/run_052_02_lca_originIC_four_baselines_static_level1.py --dry-run
python 052_lca_originIC_site_transfer/run_052_02_lca_originIC_four_baselines_static_level1.py
python 052_lca_originIC_site_transfer/run_052_00_lca_originIC_expanded_action_maskableppo.py --dry-run
python 052_lca_originIC_site_transfer/run_052_00_lca_originIC_expanded_action_maskableppo.py --smoke
python 052_lca_originIC_site_transfer/run_052_00_lca_originIC_expanded_action_maskableppo.py --formal
python 052_lca_originIC_site_transfer/run_052_01_lca_originIC_external_auto_n_rule_minimal.py --dry-run
python 052_lca_originIC_site_transfer/run_052_01_lca_originIC_external_auto_n_rule_minimal.py
python 052_lca_originIC_site_transfer/run_052_03_lca_originIC_five_scenario_figures.py --dry-run
python 052_lca_originIC_site_transfer/run_052_03_lca_originIC_five_scenario_figures.py
```

Use `--run-id <label>` for repeated auto-N threshold tests.

If `benchmark_results/052_02_lca_originIC_four_baselines_static_level1/evaluation/052_02_coverage_manifest.csv`
contains non-ok rows, inspect `052_02_management_event_audit.csv` before using the five-scenario figures.

The figure entry point defaults to:

- PPO root: `benchmark_results/052_00_lca_originIC_expanded_action_maskableppo`
- auto root: `benchmark_results/052_01_lca_originIC_external_auto_n_rule_nstd050_minimal`
- baseline root: `benchmark_results/052_02_lca_originIC_four_baselines_static_level1`


