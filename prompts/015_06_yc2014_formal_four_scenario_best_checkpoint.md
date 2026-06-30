# 015_06 YC2014 正式四情景对照与 best checkpoint 汇总

## 目的

在 YC2014 上把统一 DQN 的结果整理成正式可汇报版本。

已有证据表明：

- null、recorded、DSSAT auto 和 DQN 都可以在同一站点年份下完成模拟；
- DQN 需要采用 checkpoint selection，而不是 final model；
- seed0 和 seed1 都能在中途 checkpoint 达到高产水氮策略。

本轮目标是把这些证据合并成一张正式四情景图，并附上 seed0 / seed1 best checkpoint 汇总表，供导师和论文正文使用。

## 固定对象

- 站点年份：YC2014
- 四个正式情景：
  - null
  - recorded expert
  - DSSAT auto
  - DQN best checkpoint
- DQN 采用 best checkpoint，不采用 final model。

## DQN best checkpoint 选取规则

- seed0：使用 015_04 中产量最高的 checkpoint；
- seed1：使用 015_05 中产量最高的 checkpoint；
- 如出现并列，则优先选更早的 checkpoint。

## 图形要求

单张主图尽量包含：

- 降雨柱状图；
- 水分胁迫折线；
- 氮胁迫折线；
- 灌溉/施肥管理事件；
- 两种产量轨迹（grain 与 biomass）；
- cumulative reward 子图或折线。

风格保持前面 `Nature` 风格但不花哨，白底、黑字、对比度高。

## 输出

保存到：

```text
DSSAT_auto_validation/yc2014_formal_four_scenario_015_06/
```

输出：

- 四情景主图；
- 四情景日值 CSV；
- seed0 / seed1 best checkpoint 汇总表；
- 中文实验记录 MD；
- 如有必要，保留 seed best checkpoint 对比的单独小表。

