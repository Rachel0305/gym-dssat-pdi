# 022_10 SY2014 DAP65 action1 vs action7 受控DSSAT对照

## 1. 目的

022_09发现三个离线checkpoint在DAP65均把action7（I15/N100）排在action1（I15/N0）之前，但该经验target均值受前期历史混杂。本任务用固定前后缀、只交换DAP65动作的确定性DSSAT对照，验证该具体状态下的真实因果差异。

## 2. 固定方案

基础方案固定为022_01已验证场景 `W60_critical__N200_early`：

| DAP | 1 | 30 | 50 | 65 | 85 | 110 |
|---|---:|---:|---:|---:|---:|---:|
| Control action | 3 | 4 | 7 | 1 | 1 | 0 |
| Treatment action | 3 | 4 | 7 | 7 | 1 | 0 |

唯一差异：DAP65由action1（I15/N0）换为action7（I15/N100）。DAP65前累计I30/N200，剩余预算I90/N100，因此两个动作均原样执行，不发生裁剪或别名。

预期季节总量：Control I60/N200；Treatment I60/N300。

## 3. 冻结条件

- SY2014、IC=2、相同PDI/DSSAT4.8.0输入；
- 相同阶段窗口、初始条件、天气、土壤、品种；
- 相同DAP1/30/50/85/110动作；
- 确定性前向模拟各1次；不训练DQN；
- 保存逐日值、阶段动作、Summary匹配、产量、ET、WP、PFP和完整季节return。

## 4. 预注册判定

- **A action1在该固定状态占优**：两组均达到产量门槛，且Control的G0_raw高于Treatment；额外N100的产量增益不足抵消500 reward成本。支持后续对该状态的Q(action1)>Q(action7)排序约束。
- **B action7在该固定状态占优**：Treatment的G0_raw更高，说明额外N100在该状态下的收益足以覆盖成本，不能用经验均值纠正Q。
- **C 农学收益与reward目标分离**：Treatment产量更高但G0更低；需要明确DQN目标是资源成本加权回报而非单纯最高产量。
- **D 实现失败**：非DAP65动作、输入、预算、执行量或回放误差不一致。

若Treatment增产但不足500 kg/ha，按C报告，同时Control仍是当前reward下的最优动作。不得事后改变成本系数。

## 5. 输出

- 两方案stage/daily/summary CSV；
- 差值表、JSON；
- PNG/SVG；
- 中文记录。
