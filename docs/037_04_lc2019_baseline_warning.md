# 037_04 LC2019 基线警告说明

037_04 生成的 LC2019 五情景图目前只能作为调试图，不能作为正式汇报证据。

复核发现：

- `034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/LCA/2019/official_extension_expert/fileX.MZX` 中存在多条 official expert 灌溉/施肥计划；
- 但 DSSAT 实际使用的 `DSSAT48.INP` 只包含第一条灌溉和第一条施肥；
- `MgmtEvent.OUT` 也只记录 DAP6 一次 Fertilizer 82 kg[N]/ha 和一次 Irrigation 22.5 mm。

因此当前 LC2019 official expert 曲线不应被解释为完整专家分期方案。

此外，LC2019 的 null、recorded、auto、expert、PPO 在当前 snapshot 中最终 GWAD 均为 9807 kg/ha，且水分胁迫为 0、氮胁迫极低。该现象可能与 LC2019 初始水氮条件/年份不敏感有关，但在 official expert 基线事件确认修正前，不应将其作为 PPO 管理优越性的正式证据。

下一步应优先审计并修正基线管理事件从 `fileX.MZX` 到 `DSSAT48.INP`/`MgmtEvent.OUT` 的实际生效链路，再重建五情景图。
