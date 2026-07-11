# 020_07 HLA2010 n-step DQN seed2 稳定性复核

020_06 的 n-step seed1 在10K checkpoint达到7854 kg/ha、I120/N0并通过严格基线，因此按预先规则补做第二个独立 seed。

- 唯一算法改动：n_steps=5；
- 其他完全复用020_06/015_12：HLA2010、IC=1、9动作、baseline-relative reward、I≤120、N≤300、7天间隔和全部超参数；
- seed=2，正式50K，每5K保存checkpoint；
- 先500-step smoke，之后才运行50K；
- 按total_reward最大、并列最早选择checkpoint；
- 用原生Summary.OUT计算WP_ET、NLCM并判断严格成功；
- 不修改旧结果，不追加其他算法改动。

输出：

```text
DSSAT_auto_validation/HLA_2004/hla2010_nstep_dqn_seed2_020_07/
docs/2026-07-10_020_07_hla2010_nstep_dqn_seed2_record.md
```
