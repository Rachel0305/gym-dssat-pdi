# Project progress

## 2026-07-11 — 021_00 Benchmark Framework Refactor

- 状态：框架实现与 smoke test 已完成；框架提交 `3bec052` 已推送，正式记录待提交。
- 审计：已确认训练/评估入口、冻结 reward/action/constraint、五站点输入、历史结果和 checkpoint 边界。
- 新框架：YAML、experiment_id/config_hash、registry、dry-run、train/evaluate/report 模式、checkpoint resume、统一 CSV/Excel/PNG/SVG/MD/PPT。
- smoke：HLA2007 seed0，100 步中断后恢复至 200，13 项检查通过；重复运行成功复用。
- 未改变：冻结 DQN、reward、IC、旧脚本和旧结果。
- 下一步：先 report-only 汇总已有证据，再按 021_01–021_07 做单因素敏感性和跨年份/跨站点验证；长训练前必须 smoke。
- 阻塞：SY IC provenance、LC/SY 冻结正式训练、ET/N uptake 统一抽取。
