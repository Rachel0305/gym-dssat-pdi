# 011_02 HLA action-channel diagnosis before PPO restart

## 背景

011_01 已经验证：

- `references/rewards.py` 官方 reward 可以被加载。
- Python/gym 层能发出非零动作。
- 但非零动作没有进入 DSSAT 管理事件和最终产量：
  - `MgmtEvent.OUT` 灌溉事件数 = 0
  - `MgmtEvent.OUT` 施肥事件数 = 0
  - 2010 最终 `GWAD=6956 kg/ha`，与 null 一致

因此当前不能进入 PPO 训练。

## 核心问题

要回答：

> `env.step(action)` 中的 `amir/anfer` 到底有没有被 PDI/DSSAT 用于更新管理措施和作物生长？

不能只看 `real_action_amir/anfer` 日志，因为日志只能说明 Python 层动作被发出，不等于 DSSAT 执行。

## 低成本诊断目标

只做 forward/action-channel 诊断，不训练。

1. 固定同一个 2010 input。
2. 运行两个 gym episode：
   - null action：全程 `amir=0, anfer=0`
   - forced action：DAP 1 `anfer=165`，DAP 49/70/95 `amir=10`
3. 对比两者的 PDI/DSSAT raw 输出：
   - `PlantGro.OUT`
   - `PlantN.OUT`
   - `SoilWat.OUT`
   - `MgmtEvent.OUT`
   - `Summary.OUT`
4. 判断 forced action 是否造成以下任一变化：
   - 管理事件增加；
   - `GWAD/CWAD` 改变；
   - `WSPD/NSTD/SWFAC/NSTRES` 曲线改变；
   - `TOTIR/TOFER` 或相关累积量改变。

## 通过标准

动作通道通过必须满足至少一条强证据：

- `MgmtEvent.OUT` 出现对应灌溉或施肥事件；或
- forced action 与 null action 的最终产量/生物量/胁迫曲线出现明确差异，且差异方向合理。

如果 forced action 与 null raw 输出完全一致，则说明当前 gym action channel 对 DSSAT 生长模拟无效，必须先修 action channel，不能训练 PPO。

## 安全规则

- 不训练 PPO。
- 不改 Docker/site-packages。
- 不覆盖旧结果。
- 只在 `DSSAT_auto_validation/HLA_2004/hla2010_2015_official_reward_restart/action_channel_diagnosis` 下保存新输出。
- 保存诊断代码、CSV、raw 文件快照和结论 README。

