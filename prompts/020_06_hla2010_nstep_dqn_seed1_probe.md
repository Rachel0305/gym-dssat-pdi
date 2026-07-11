# 020_06 HLA2010 n-step DQN seed1 单变量探针

## 历史核对

- 旧项目曾在 HLA2015、不同 reward 和动作线路中测试 n-step=5。
- 当前 HLA2010 的 015_12 baseline-relative reward、9动作、IC=1 框架尚未测试 n-step。
- 因此本轮不是重复旧实验。

## 唯一改动

把标准 DQN 的 `n_steps` 从1改为5，使终止产量奖励更快向前期动作传播；其余全部保持015_12不变：

- HLA2010、seed1、IC=1、相同天气/土壤/品种/MZX；
- baseline-relative reward：终端 `max(0, GWAD_final - GWAD_null)` 减水氮成本；
- 9动作，I∈{0,15,30}，N∈{0,50,100}；
- I≤120、N≤300，单次上限，最小操作间隔7天；
- 学习率、buffer、batch、gamma、exploration等全部沿用015_12。

## 执行顺序

1. 500-step smoke：只检查兼容性、动作传输、MgmtEvent、输出和内存；不判断策略。
2. smoke通过后运行50K，每5K保存 checkpoint。
3. checkpoint仍按 `total_reward` 最大、并列取最早选择。
4. 用Summary.OUT计算WP_ET、NLCM并与标准DQN seed1比较。

若 n-step seed1改善，再用相同配置复核 seed2；若不改善，关闭“单独n-step”路线，不继续堆步数或seed。

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed1_020_06/
docs/2026-07-10_020_06_hla2010_nstep_dqn_seed1_record.md
```
