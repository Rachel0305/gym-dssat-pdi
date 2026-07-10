# 018_03 五站点官方农技推广 expert baseline 扩展

## 背景

导师提出：论文中的 expert baseline 应参考全国农技推广中心推文/ PDF《玉米大豆水肥一体化单产提升技术方案》，而不是仅使用历史实测/农民记录管理。此前的 `recorded expert` 应更准确命名为 `recorded/farmer practice`。

018_02 已在 HLA2010 和 YC2014 做了低成本小试，确认新增 `official extension expert fixed DAP` 情景可运行，且不会推翻现有 DQN 叙事。

## 本轮目标

将官方推广 expert baseline 扩展到五个站点的代表年份：

- HLA2010：东北及长城沿线春玉米区，PDF 表 1。
- SY2014：东北及长城沿线春玉米区，PDF 表 1。
- YC2014：华北黄淮和汾渭平原夏玉米区，PDF 表 3。
- FQ2016：华北黄淮和汾渭平原夏玉米区，PDF 表 3。
- LC2010：华北黄淮和汾渭平原夏玉米区，PDF 表 3。

如果某个站点输入链路尚未完全稳定，允许记录为失败，不要为了出结果临时改输入或改参数。

## 严格限制

- 不训练 DQN。
- 不修改 DQN 奖励函数。
- 不修改原始 `.MZX`、`.WTH`、`.SOL`、`.CUL`、`.jinja2` 输入文件。
- 不覆盖已有结果。
- 只新增 `extension_expert_fixed_dap` replay 情景。
- 每个站点先只跑一个代表年份，作为导师沟通前的小闭环。
- 如果单次灌溉超过 50 mm，拆成相邻日事件。
- 所有输出写入 `DSSAT_auto_validation/extension_expert_baseline_018_03/`。

## 固定 DAP 映射

暂时不用实测生育期，也不用 DSSAT 事后模拟生育期，避免引入事后信息。

东北春玉米区（HLA/SY，表 1）：

- DAP 0：播种/基肥
- DAP 30：小喇叭口期
- DAP 50：大喇叭口期
- DAP 65：抽雄散粉期
- DAP 85：灌浆初期
- DAP 110：乳熟末期

华北黄淮夏玉米区（YC/FQ/LC，表 3）：

- DAP 7：出苗水
- DAP 30：小喇叭口期
- DAP 45：大喇叭口期
- DAP 60：抽雄散粉期
- DAP 80：灌浆期
- DAP 100：乳熟期

## 输出

至少输出：

- `018_03_extension_expert_schedule.csv`
- `018_03_extension_expert_daily.csv`
- `018_03_extension_expert_events.csv`
- `018_03_extension_expert_summary.csv`
- `018_03_clean_multisite_comparison_with_extension_expert.csv`
- `figures/018_03_extension_expert_summary.png`
- `docs/2026-07-09_018_03_multisite_extension_expert_baseline_record.md`

## 重点解释

本轮不是为了证明 DQN 最优，而是为了回答导师的新 baseline 要求：

1. 官方推广 expert baseline 能否在现有 gym/PDI-DSSAT 链路中稳定 replay？
2. 与已有 null / recorded-farmer / DSSAT auto / DQN 相比，官方推广 expert 是高投入高产，还是低投入高效？
3. 当前 DQN 结果是更像“追平高产平台并节约资源”，还是仍需要奖励函数改造？

## 执行环境

在指定 Docker 容器中执行：

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_extension_expert_baseline_018_03.py"
```

