# 020_09 HLA2010 n-step DQN → HLA2015 跨年份迁移

## 历史核对

- 015_17 已完成标准 baseline-relative DQN 的 HLA2010→HLA2015 迁移。
- 本轮不重复标准DQN迁移，也不重新训练 HLA2015。
- 当前 n-step HLA2010 已有三个通过严格基线的 checkpoint：seed0/30K、seed1/10K、seed2/20K；此前没有用 n-step 模型做 HLA2015 迁移。

## 设置

- 训练年：HLA2010；测试年：HLA2015。
- 只加载 n-step 已训练模型，不继续训练。
- 测试模型：seed0/30K、seed1/10K、seed2/20K。
- HLA2015 使用自己的 IC=1 输入、null baseline和环境；评估 reward 使用 HLA2015 null产量。
- 对照保留已有 HLA2015 null、官方expert、DSSAT auto和local DQN。

## 判定

分别报告：是否明显高于null、是否接近/超过auto和expert、灌溉和施氮是否合理、三个迁移seed之间是否一致。迁移成功不等于跨站点泛化。

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2010_nstep_to_hla2015_transfer_020_09/
docs/2026-07-10_020_09_hla2010_nstep_to_hla2015_transfer_record.md
```
