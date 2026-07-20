# 026_09 HLA 冻结 SY 阶段型 PPO 跨站点验证记录

## 状态

`partial / stopped_before_first_policy_action`

## 已完成

- 已写预注册 prompt 和独立评估脚本。
- 已完成 HLA2010 seed0 的输入准备与环境 reset。
- 未训练模型，未覆盖既有结果，冻结模型未修改。

## Smoke 失败

第一次策略动作前，wrapper 检查发现 HLA 原始观测为 24 维，而 SY 冻结模型和 021_24 scaler 要求 25 维：

```text
ValueError: Expected wrapper observation dimension 25, got (24,)
```

进一步检查表明，差异来自土壤含水量剖面：SY 为 9 层，HLA 当前环境为 8 层。该 smoke 执行了 0 个 PPO 动作，因此**不构成 HLA 策略成功或失败的科学结果**。

## 决策

没有使用补零、复制最深层或删除 scaler 特征等任意捷径。正式 HLA 15 季评估暂停，先执行 026_10 五站点观测维度与物理土层深度审计。只有在预注册并验证科学可解释的深度映射 adapter 后，才允许恢复跨站点评估。

