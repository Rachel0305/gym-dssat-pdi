# 054 HLA lowIC site-transfer workflow

This folder keeps the HLA lowIC transfer entry points together.  It follows the
same four-step layout used by the later LCA workflows:

1. `054_02` rebuilds the four non-PPO baselines with the static level-1
   management correction.
2. `054_00` trains/evaluates the expanded-action MaskablePPO.
3. `054_01` runs DSSAT native auto-irrigation plus minimal external auto-N.
4. `054_03` builds the five-scenario daily figures and seasonal bar charts.

Run from `/workspace/src` in the container:

```bash
python 054_hla_lowIC_site_transfer/run_054_02_hla_lowIC_four_baselines_static_level1.py --dry-run
python 054_hla_lowIC_site_transfer/run_054_02_hla_lowIC_four_baselines_static_level1.py

python 054_hla_lowIC_site_transfer/run_054_00_hla_lowIC_expanded_action_maskableppo.py --dry-run
python 054_hla_lowIC_site_transfer/run_054_00_hla_lowIC_expanded_action_maskableppo.py --smoke
python 054_hla_lowIC_site_transfer/run_054_00_hla_lowIC_expanded_action_maskableppo.py --formal

python 054_hla_lowIC_site_transfer/run_054_01_hla_lowIC_external_auto_n_rule_minimal.py --dry-run
python 054_hla_lowIC_site_transfer/run_054_01_hla_lowIC_external_auto_n_rule_minimal.py

python 054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py --dry-run
python 054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py
```

Use `--run-id <label>` for repeated auto-N threshold tests.

The standardized 054 defaults are:

- PPO config: `configs/054_00_hla_lowIC_expanded_action_maskableppo.json`
- auto config: `configs/054_01_hla_lowIC_external_auto_n_rule_nstd050_minimal.json`
- four-baseline config: `configs/054_02_hla_lowIC_four_baselines_static_level1.json`
- PPO root: `benchmark_results/054_00_hla_lowIC_expanded_action_maskableppo`
- auto root: `benchmark_results/054_01_hla_lowIC_external_auto_n_rule_nstd050_minimal`
- static four-baseline root: `benchmark_results/054_02_hla_lowIC_four_baselines_static_level1`

Note: HLA keeps `station_code=HLA` and `site=HLA` in configs, but its DSSAT
input directory is `HL/` and the source template is
`CNHL0701_corrected_IC123.MZX`.
