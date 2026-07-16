# 022_11 SY2014 DAP65 action1/action7 三前缀一致性验证

## 1. 依据

022_10在 `W60_critical__N200_early` 固定状态下证明：DAP65额外N100仅增产0.106 kg/ha，G0下降499.894，action1在联合目标上明显优于action7。为避免用单一状态设计排序约束，本任务只补两个独立水分前缀，并与022_10合并形成固定的三前缀判定。

## 2. 两个新增前缀

### Prefix B：W75 uniform-pre90背景

- Control：[4,4,7,1,1,0]
- Treatment：[4,4,7,7,1,0]
- DAP65前：I45/N200；最终预期I75/N200 vs I75/N300。

### Prefix C：W120 critical背景的固定前后缀变体

- Control：[3,5,8,1,2,0]
- Treatment：[3,5,8,7,2,0]
- DAP65前：I60/N200；最终预期I105/N200 vs I105/N300。
- 该方案借用 `W120_critical__N200_early` 的前缀与后缀，但为保持action1/action7同灌溉档对照，DAP65 Control固定为action1，因此不声称完整复现原W120场景。

两对唯一差异均为DAP65 action1（I15/N0）与action7（I15/N100）；剩余氮预算均为N100，不发生裁剪。

## 3. 一致性门槛

固定材料性阈值：`Treatment−Control G0_raw ≤ −250`。250来自额外N100对应reward成本500的一半，不依据结果选取。

- **A 3/3一致**：022_10和两个新增前缀均满足两组产量达到11077 kg/ha，且ΔG0≤−250。允许下一步进行一次纯离线pairwise-loss与MC-loss梯度冲突检查。
- **B 不一致/上下文依赖**：任一前缀不满足上述条件，立即停止pairwise约束路线；不得补第4个前缀。
- **D 实现失败**：非DAP65动作不一致、预算裁剪、输入或Summary匹配失败。

不以生物量变化直接证明“库容饱和”；只能作为机制解释线索。

## 4. 边界与输出

- 仅4次确定性DSSAT前向模拟；DQN训练0；
- 保存两前缀summary/stage/daily、差值、与022_10合并判定表、PNG/SVG、JSON和中文记录；
- 不修改reward、动作、IC、品种或判据。
