# 011_03 HLA linked-management action-channel test

## 背景

011_01/011_02 发现：Python/gym 层动作已经发出，但 DSSAT `PlantGro.OUT` 完全不变。

检查源码后发现关键原因：

- `DssatPdi.__init__` 中，`mode='all'` 会设置：
  - `self.irrig = 'L'`
  - `self.ferti = 'L'`
- 但这只有在 fileX 模板含有 Jinja 占位符时才会生效。
- 当前 HLA MZX 是静态 DSSAT 文件，不含 `{{irrig}}` / `{{ferti}}` 占位符。
- 因此实际 `pdi_tmp_snapshot/fileX.MZX` 仍然是：
  - `IRRIG=R`
  - `FERTI=R`

也就是说，PDI action 通过 socket 发给了客户端，但 DSSAT 管理模式没有进入 linked/action 模式。

## 本轮目标

只做低成本 forward 诊断，不训练 PPO。

在同一套 2010 input 下比较四组：

1. `null_action`：`IRRIG=R, FERTI=R`，action 全 0；
2. `forced_action`：`IRRIG=R, FERTI=R`，强制 action；
3. `linked_null_action`：`IRRIG=L, FERTI=L`，action 全 0；
4. `linked_forced_action`：`IRRIG=L, FERTI=L`，强制 action。

强制 action：

- DAP 1：`anfer=165`
- DAP 49/70/95：`amir=10`

## 判断标准

如果 `linked_forced_action` 相比 `linked_null_action` 出现：

- `MgmtEvent.OUT` 管理事件变化；或
- `PlantGro.OUT` 的 `GWAD/CWAD/WSPD/NSTD` 发生变化；

则说明问题不是 socket/action 发送失败，而是 HLA 静态 MZX 没有正确切换到 PDI linked management。

如果 `L/L` 后仍然完全无响应，则需要继续查 PDI 侧 action 变量是否真正绑定到 DSSAT 管理模块。

## 安全规则

- 不训练。
- 不改 Docker/site-packages。
- 不覆盖旧结果。
- 只保存诊断输出。

