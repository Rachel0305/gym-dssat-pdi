# 019_04 runtime leaching observation probe

## 目的

在不训练的前提下，确认 `cleach` / `tleachd` / `cnox` 是否能从 PDI/gym-DSSAT 的运行时 observation 中被读取出来，并与 DSSAT 原始输出 `Summary.OUT` / `SoilNi.OUT` 对照。

## 背景

019_03 已确认：

- `dssat-pdi.yml` 中已有 `CLeach -> cleach` 与 `TLeachD -> tleachd`。
- `Summary.OUT` 中已有 `NLCM`。
- `SoilNi.OUT` 中已有 `NLCC`。
- 当前主线 DQN reward 和日值 CSV 尚未稳定使用这些变量。

因此本轮不改模板、不训练，只做运行时读取验证。

## 实验设计

站点年份：FQ2016。

两个确定性回放情景：

- `I0_N0`：不灌溉、不施氮。
- `I120_N300`：按固定 DAP 分配 120 mm 灌溉和 300 kg/ha 施氮。

每一步记录：

- `cleach`
- `tleachd`
- `cnox`
- `grnwt`
- `topwt`
- `swfac`
- `nstres`
- 灌溉/施肥动作

结束后复制 PDI 快照，并解析：

- `Summary.OUT` 的 `NLCM`
- `SoilNi.OUT` 的 `NLCC`

## 判断规则

如果运行时 daily CSV 中存在并更新 `cleach/tleachd/cnox`，且最终值与原始 DSSAT 输出同向、数值合理，则下一步可以写 leaching-aware reward smoke test。

如果运行时 observation 没有这些变量，但快照中有，说明需要修改 wrapper/observation 读取逻辑。

如果快照中也没有，才回到模板层面处理。
