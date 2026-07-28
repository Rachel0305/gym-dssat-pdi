# 037_05：DSSAT 管理事件生效链路预检

## 背景

037_04 生成 LC2019 五情景图后发现异常：`fileX.MZX` 中 official expert 看起来有多条灌溉/施肥计划，但 DSSAT 实际运行使用的 `DSSAT48.INP` 和 `MgmtEvent.OUT` 只包含/执行了第一条。  

这说明后续不能只看渲染后的 `fileX.MZX` 或最终图，而必须在训练、汇总、绘图之前检查：

> 计划管理事件是否真实进入 DSSAT，并被 DSSAT 执行。

## 任务

审计当前统一 IC=1 四基线结果：

`benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots`

对每个站点、年份、情景检查：

1. `fileX.MZX` 中当前 treatment 对应的灌溉/施肥计划事件；
2. `DSSAT48.INP` 中实际写入的灌溉/施肥事件；
3. `MgmtEvent.OUT` 中实际执行的灌溉/施肥事件；
4. 三者的事件数量和总量是否一致。

## 判定

- 若 `fileX.MZX` 计划事件数大于 `DSSAT48.INP` 实际事件数，标记为 `event_chain_issue`；
- 若 `DSSAT48.INP` 与 `MgmtEvent.OUT` 事件数不同，标记为 `event_chain_issue`；
- 若任意两层的灌溉或施氮总量差异超过 0.2，标记为 `event_chain_issue`；
- 若关键文件缺失，标记为 `missing_required_file`；
- 任意异常 snapshot 不得直接进入正式五情景图和指标比较。

## 输出

- 每个 snapshot 的事件链路明细 CSV；
- 异常 snapshot 清单 CSV；
- 中文实验记录 MD。

## 停止线

本任务只读文件，不修改 DSSAT 输入，不重跑 DSSAT，不训练模型。若发现异常，后续先修正管理事件生效链路，再重建基线/训练/绘图。
