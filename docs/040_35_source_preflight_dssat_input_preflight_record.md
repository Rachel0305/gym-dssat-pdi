# 040_35_source_preflight DSSAT 输入预检查记录

- 输入根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：`SYA`
- 预期 IRRIG：`R`
- 预期 FERTI：`R`
- 指定权威 MZX：`None`
- 是否通过：`False`
- 问题数：`1`

## 输出

- 输入 manifest：`benchmark_results/000_dssat_input_preflight_gate/040_35_source_preflight/tables/input_manifest.csv`
- IC/管理模式检查：`benchmark_results/000_dssat_input_preflight_gate/040_35_source_preflight/tables/mzx_ic_management_checks.csv`
- 问题清单：`benchmark_results/000_dssat_input_preflight_gate/040_35_source_preflight/tables/preflight_issues.csv`
- manifest 对比：`benchmark_results/000_dssat_input_preflight_gate/040_35_source_preflight/tables/manifest_compare.csv`
- JSON 摘要：`benchmark_results/000_dssat_input_preflight_gate/040_35_source_preflight/preflight_result.json`

## 问题

- SYA: CNSY1201(1).MZX treatment 2 IC=0 not enabled or missing IC block
