# 021_43 SY2014 reward与导师目标一致性离线审计

## 目的

检查021_41/021_42中低氮但低于expert产量的checkpoint，是否因当前冻结reward `max(0,Y-null)-I-5N`反而获得更高分，从而使DQN在优化reward时偏离“产量不低于expert且提高水氮效率”的严格目标。

## 边界

- 纯计算，复用seed0/1/2已有checkpoint结果；
- 不训练、不调用DSSAT、不修改reward；
- 对每个在线seed分别比较reward最高checkpoint与expert gate；
- 计算通过gate的候选与失败高reward候选之间的氮成本盈亏平衡值，仅用于解释，不据此现场改系数。

## 判据

- A：seed1和seed2的reward最高checkpoint均未通过expert gate，但各自存在通过gate且reward更低的checkpoint；系统性目标错位；
- B：仅一个seed出现；
- C：均未出现。

