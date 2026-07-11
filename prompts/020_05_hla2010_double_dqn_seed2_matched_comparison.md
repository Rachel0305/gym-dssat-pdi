# 020_05 HLA2010 Double DQN seed2 严格单变量复核

## 历史核对

- 020_04 已完成当前正式框架的 Double DQN seed1 50K；固定规则选中30K，HWAM=7854、I=120、N=0，并通过严格基线判定。
- 旧项目的 Double DQN 结果属于 HLA2015、旧 economic-reward 和4动作线路，不能替代当前 HLA2010 seed2。
- 当前没有 HLA2010 Double DQN seed2 结果，因此本轮不是重复实验。

## 固定设置

- 站点年份：HLA2010，IC=1，更新后的品种参数和同一输入模板。
- 算法：`CustomDoubleDQN`，唯一算法改动为 Double DQN TD target。
- Seed：2；正式训练50K，每5K保存 checkpoint。
- Reward、9动作、I≤120、N≤300、单次上限、7天最小间隔和全部超参数完全复用015_12/020_04。
- checkpoint选择：`total_reward`最大；并列取最早；不按产量或图形人工挑选。

## 执行顺序

1. 先运行500-step smoke，检查容器、动作链、MgmtEvent、输出和内存。
2. smoke通过后运行50K正式训练。
3. 用DSSAT原生Summary.OUT计算WP_ET、NLCM，并与auto、official expert及标准DQN seed1/Double DQN seed1对比。

## 判定

如果 seed2 也达到产量约7854、WP_ET不低于基线且水氮投入与淋洗不高，才可以初步称 Double DQN 在 HLA2010 有跨seed稳定迹象；否则只能称 seed1 的单seed改善。

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2010_double_dqn_seed2_020_05/
docs/2026-07-10_020_05_hla2010_double_dqn_seed2_record.md
```
