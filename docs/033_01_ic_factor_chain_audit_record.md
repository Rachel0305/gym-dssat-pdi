# 033_01 IC 因子启用链条审计记录

## 任务性质

本任务只读取现有模板和结果快照，审计 DSSAT treatment 行中 `IC` 因子是否启用；不训练、不跑 DSSAT、不修改代码。

## 审计输出

- 逐文件表：`benchmark_results/033_01_ic_factor_chain_audit/033_01_ic_factor_rows.csv`
- 汇总表：`benchmark_results/033_01_ic_factor_chain_audit/033_01_ic_factor_summary.csv`
- 问题文件表：`benchmark_results/033_01_ic_factor_chain_audit/033_01_problem_files.csv`

## 总体计数

- 审计文件数：771
- `IC=0`：759
- `IC>0` 且能对应初始条件块：12
- `IC>0` 但找不到对应初始条件块：0
- 解析问题：0

## 按文件组汇总

| 文件组 | 文件数 | IC=0 | IC有效启用 | IC非零但缺块 | IC取值 |
|---|---:|---:|---:|---:|---|
| 031_35_four_baseline_snapshots | 384 | 384 | 0 | 0 | 0 |
| 031_36_auto_baseline_snapshots | 96 | 96 | 0 | 0 | 0 |
| 032_22_maskableppo_rendered | 250 | 250 | 0 | 0 | 0 |
| 033_00_derived_inputs | 36 | 24 | 12 | 0 | 0;1 |
| source_templates | 5 | 5 | 0 | 0 | 0 |

## 源模板情况

源模板共 5 个；IC=0 的源模板 5 个，IC 非零且可对应 INITIAL CONDITIONS 的源模板 0 个。

## 初步结论

- 发现大量文件的 treatment `IC` 因子为 0；这些文件即使包含 `*INITIAL CONDITIONS` 表，也不会通过 treatment 因子启用该初始条件。
- 存在 IC 非零且能对应 `*INITIAL CONDITIONS` treatment id 的文件，说明审计脚本能识别已启用 IC 的分支。

## 下一步建议

若主流程渲染输入确认为 `IC=0`，应先修复安全渲染函数，使其显式启用当前模板中的 `*INITIAL CONDITIONS`；修复后先做小规模 smoke，确认渲染后的 treatment 行 `IC=1` 且 WSPD 对初始水分有响应，再决定全量重跑范围。
