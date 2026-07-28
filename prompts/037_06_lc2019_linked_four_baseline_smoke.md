# 037_06：LC2019 linked-action 四基线可信烟测

## 背景

037_05 发现 034_00 的手工基线存在管理事件链路异常：部分 `official_extension_expert` / `recorded_farmer_template` 在 `fileX.MZX` 中计划了多次灌溉/施肥，但 `DSSAT48.INP` 与 `MgmtEvent.OUT` 只执行第一条或明显更少事件。

034_03 已验证 linked management 动态动作通道能够把 PPO 的外部动作真实送进 DSSAT。因此本任务先用 LC2019 做最小修复烟测：不再依赖静态 `fileX.MZX` 管理表执行手工基线，而是在每日 step 中按固定计划发送精确物理动作。

## 任务

只跑 LCA/LC 2019 一个站点年份的四情景：

1. `null`：每日 no-op；
2. `recorded_farmer_template`：按 027_05 recorded farmer 模板计划日逐日发送动作；
3. `official_extension_expert`：按 extension expert 计划日逐日发送动作；
4. `dssat_auto`：保留 DSSAT 自动管理。

## 核心判据

对手工基线：

- 计划事件数；
- step 发送事件数；
- `MgmtEvent.OUT` 实际执行事件数；
- 计划总灌溉/施氮量；
- `MgmtEvent.OUT` 实际总灌溉/施氮量；
- `Summary.OUT` 汇总总灌溉/施氮量。

若三者数量和总量一致，说明该情景的管理措施真实进入 DSSAT。

对 `dssat_auto`：

- 不要求 `fileX.MZX` 存在手工计划事件；
- 只记录 `MgmtEvent.OUT` 和 `Summary.OUT` 中 DSSAT 自动执行的水肥事件。

## 输出

- LC2019 四情景日值表；
- LC2019 四情景 summary；
- 管理事件链路审计表；
- 中文实验记录。

## 停止线

本任务只验证一个站点年份的可信基线重建方式。若 LC2019 仍不能做到 step 发送事件与 `MgmtEvent.OUT` 一致，不进入全站点/全年份重建。
