# 028_10 FQ2016 缺失 seed1/2 阶段型 MaskablePPO 补齐

## 目的

027_07 因旧 primary 停止规则只运行了 FQ2016 seed0。导师当前汇报规则改为“产量、WP_ET、PFP_N 至少一项严格超过四基线，其余两项报告接近程度”。本任务不重复 seed0，只补同一冻结配置的 seed1/2，以判断 FQ 是否存在可迁移的已训练 PPO checkpoint。

## 冻结配置

完全复用 027_07 FQ readiness、scaler、reward、网络、240 stage steps、checkpoint 0/60/120/180/240、确定性评估和按 episode reward 选模规则。不得修改 reward、输入、IC、动作、mask 或 checkpoint 选择。

## Smoke 依据

027_07 FQ seed0 已通过同一代码路径的全部工程检查（240 steps、4 learn calls、5 checkpoints、零 invalid action、哈希不变），因此它是本次不重复执行的既有 smoke 证据。

## 执行

- 串行 seed1 后 seed2；不得并行。
- 每 seed 240 stage steps。
- 保存全部 checkpoint、训练季节、评估动作、选择结果和失败记录。
- 与既有 seed0 合并报告，但不改写 027_07。

## 判定

只使用训练后 checkpoint 60/120/180/240 的预注册选模结果；checkpoint0 即便偶然通过也不能冒充训练成功。

