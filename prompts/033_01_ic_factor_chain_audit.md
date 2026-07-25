# 033_01 IC 因子启用链条审计 prompt

## 背景

033_00 初始土壤水分敏感性审计发现：

- 在当前渲染模板的 `IC` 因子保持原值时，降低 `*INITIAL CONDITIONS` 中的 `SH2O` 对 WSPD 和产量没有影响；
- 将 treatment 行的 `IC` 因子强制设为 1 后，降低 `SH2O` 会显著改变 WSPD 和产量；
- 这提示主流程可能没有真正启用 DSSAT 的 `*INITIAL CONDITIONS`。

本任务只做文件链条审计，不训练、不跑 DSSAT、不修改代码。

## 目标

系统检查以下文件中的 `*TREATMENTS` 因子设置：

1. `my_data/UFGA8201-*.jinja2` 源模板；
2. 当前 PPO/DQN 相关结果目录中的已渲染输入；
3. 033_00 审计任务生成的 current/force 分支派生输入；
4. 已补齐四基线/auto 基线的快照输入。

重点提取：

- treatment 行中的 `IC` 因子；
- `MI/MF` 因子；
- `*INITIAL CONDITIONS` 中实际存在的 IC treatment id；
- `IC` 是否为 0；
- `IC` 非 0 时是否能对应到现有 `*INITIAL CONDITIONS` treatment id。

## 输出

输出目录：

```text
benchmark_results/033_01_ic_factor_chain_audit/
```

需要生成：

1. `033_01_ic_factor_rows.csv`：逐文件审计表；
2. `033_01_ic_factor_summary.csv`：按文件组汇总；
3. `033_01_problem_files.csv`：存在明显问题的文件；
4. `docs/033_01_ic_factor_chain_audit_record.md`：中文实验记录。

## 判读规则

- 若源模板和主流程渲染输入均为 `IC=0`，而 `*INITIAL CONDITIONS` 存在，则说明当前主结果大概率没有启用初始土壤水氮条件；
- 若 033_00 的 `force_ic1` 分支为 `IC=1` 且 current 分支为 `IC=0`，则说明 033_00 的敏感性结论是由 IC 因子启用差异造成的；
- 若某些文件 `IC>0` 但找不到对应的 `*INITIAL CONDITIONS` treatment id，则标记为 `ic_nonzero_missing_block`；
- 本任务不决定最终重跑范围，只给出是否需要修复渲染代码和重跑结果的证据。

## 约束

- 不修改 `my_data/` 原始模板；
- 不修改训练脚本；
- 不删除或覆盖既有结果；
- 不跑训练；
- 不跑 DSSAT；
- 审计脚本、CSV 和 MD 记录均保留。
