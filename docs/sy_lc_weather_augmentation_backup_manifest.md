# SYA / LCA 天气增强实验备份清单

分支：`codex/sya-forecast-freeze-2026-08-16`；remote：`git@github.com:Rachel0305/gym-dssat-pdi.git`。归档范围共 78 个文件（约 1371.6 KiB）：其中 76 个将在本次提交中新增，`scripts/build_dssat_cli.py` 与 `src/weather_preprocess.py` 已由 Git 跟踪且未修改。另列 23 个受保护 WTH 输入作为哈希参照，不新增提交。

SHA256、生成任务、是否关键、文件大小、Git 原有 tracked/untracked/ignored 状态及 force-add 标志以 `backup_manifest.json` 为准。清单自身因自引用无法写入自身 SHA256。

| 类别 | 文件数 | Git 处理 |
|---|---:|---|
| 020_closeout | 6 | 逐项暂存 |
| audit_or_validation_code | 14 | 逐项暂存 |
| audit_report | 8 | 逐项暂存 |
| machine_evidence | 44 | 逐项暂存 |
| protected_source_WTH | 23 | 只读参照，不新增提交 |
| task_prompt | 6 | 逐项暂存 |

关键报告：
- `docs/sy_lc_random_weather_source_audit_015.md`
- `docs/provenance_resolution_report.md`
- `docs/sy_lc_wgen_readiness_review.md`
- `docs/lca_cli_parameter_audit_016.md`
- `docs/lca_wgen_window_review.md`
- `docs/sya_train_only_fill_sensitivity_review.md`
- `docs/sya_cli_parameter_audit_018.md`
- `docs/sya_wgen_readiness_review_019.md`

机器 gate / QC：
- `results/lca_cli_parameter_audit_016/parameter_qc.json`
- `results/sy_lc_random_weather_015/final_gate.json`
- `results/sy_lc_random_weather_015/wgen_readiness_review/methodology_gate.json`
- `results/sy_lc_weather_augmentation_closeout/freeze_status.json`
- `results/sya_cli_parameter_audit_018/parameter_qc.json`
- `results/sya_wgen_readiness_review_019/wgen_readiness_gate.json`

候选 CLI：
- `results/lca_cli_parameter_audit_016/CNLC_2005_2013_candidate.CLI`
- `results/sya_cli_parameter_audit_018/CNSY_2005_2013_train_only_candidate.CLI`

源 WTH、PPO checkpoint、原始 XLS 以及无关实验结果不因本次归档而添加。忽略规则下的 CSV/CLI 等，只对本 JSON 明确 `include_in_git=true` 且 `force_add_required=true` 的小型文件逐项使用 `git add -f`。
