# 031_40 phase freeze and GitHub backup

## Purpose

冻结当前自由时序 MaskablePPO 阶段性结果，并上传 GitHub 备份，作为后续参数优化和模型改进前的检查点。

## Scope

Include:

- PPO model zip files from SY and four-site checkpoint-selection experiments.
- Evaluation CSVs, selected candidate CSVs, daily output CSVs, station summary tables, and final figures.
- Prompt, record, config, and script files required to understand or reproduce the analysis flow.
- Representative five-scenario daily plot package for LCA2010.

Exclude:

- DSSAT `runs/` snapshots and copied input folders.
- Runtime train/eval temporary directories.
- Cache files and environment artifacts.
- Unrelated proposal material edits.

## Rationale

The excluded directories are large and are mostly replay/runtime artifacts. The committed material preserves the scientific result tables, model checkpoints, daily decision traces, figures, prompts, and records needed for review and project migration.

