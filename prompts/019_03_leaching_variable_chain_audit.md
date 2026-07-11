# 019_03 leaching variable chain audit

## 目的

核查当前项目中氮淋洗变量是否已经从 DSSAT/PDI 输出进入 gym-DSSAT 状态链路，以及是否可以作为后续 DQN reward 的惩罚项。

## 背景

当前 DQN 主线奖励函数主要是：

```text
reward = max(0, final_yield - local_null_yield) - water_cost * irrigation - nitrogen_cost * nitrogen
```

文献奖励形式中常见 `N_leach` 惩罚项，但本项目尚未确认该变量是否在当前 PDI/gym-DSSAT 链路中稳定可用。因此本轮不训练、不改 reward，只做离线审计。

## 审计问题

1. 现有 PDI 快照 `dssat-pdi.yml` 是否已经声明并发送 `CLeach` / `TLeachD`？
2. 现有 DSSAT 输出 `Summary.OUT` / `SoilNi.OUT` 是否包含氮淋洗相关字段？
3. 这些字段是否随高氮、高水组合出现合理变化？
4. 当前 DQN 训练与评估 CSV 是否已经保存 `cleach` / `tleachd`？
5. 是否需要立刻改模板并重训？

## 执行原则

- 不运行训练。
- 不修改原始模板、MZX、WTH、SOL、CUL。
- 只读取现有输出和代码。
- 保存 CSV 和中文实验记录。

## 预期判断

如果 `dssat-pdi.yml` 已经有 `CLeach -> cleach`，且 `Summary.OUT`/`SoilNi.OUT` 中的淋洗字段能随管理措施合理变化，则说明不一定需要先手动改模板；下一步应先在 reward wrapper 和日值记录中读取 `cleach`，再做低步长 smoke test。

如果这些字段不存在或不稳定，则需要回到 PDI 模板/输出设置层面补字段。
