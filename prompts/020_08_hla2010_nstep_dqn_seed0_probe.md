# 020_08 HLA2010 n-step DQN seed0 三seed完整复核

## 历史核对

- 当前 n-step seed1（020_06）和 seed2（020_07）均已通过严格基线。
- 当前没有 HLA2010 n-step seed0 结果；旧 n-step 试验属于其他年份和 reward，不能替代本轮。

## 设置

- 只改变 `n_steps=1 -> 5`；其余完全复用015_12和020_06/020_07。
- HLA2010、IC=1、seed=0、50K，每5K checkpoint。
- reward、9动作、I≤120、N≤300、单次上限和7天间隔全部不变。
- 先500步 smoke，通过后再正式50K。
- checkpoint按total_reward最大、并列取最早；之后核算原生WP_ET、NLCM和严格基线。

## 目的

补齐 n-step 的第三个 seed，判断两个已成功 seed 是否构成三seed稳定证据，不修改旧结果。

输出：

```text
DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed0_020_08/
docs/2026-07-10_020_08_hla2010_nstep_dqn_seed0_record.md
```
