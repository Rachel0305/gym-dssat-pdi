# 049 HLA originIC site-transfer workflow

This folder keeps the HLA transfer entry points together.

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

Use `--run-id <label>` for repeated auto-N threshold tests.

The figure entry point defaults to:

- PPO root: `benchmark_results/049_00_hla_originIC_expanded_action_maskableppo`
- auto root: `benchmark_results/049_01_hla_originIC_external_auto_n_rule_nstd050_minimal`
- baseline root: `benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild`

