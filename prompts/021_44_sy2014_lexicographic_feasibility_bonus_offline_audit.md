# 021_44 SY2014 产量可行性优先reward离线审计

## 候选统一公式

在原reward上仅于终端增加：

`feasibility_bonus = B * 1[Y >= local_official_expert_yield]`

其中`B = max seasonal resource cost = water_cost*I_budget + nitrogen_cost*N_budget = 1*120 + 5*300 = 1620`。

该数值由冻结成本和预算自动推导，不按SY结果调参；跨站点使用同一公式，但expert产量门槛取各站点年份自己的官方expert基准。

## 目的与边界

只对021_41/42已有12个checkpoint反事实重新计分，检查每个在线seed的最高分checkpoint是否转为expert-gate可行策略。不训练、不调用DSSAT、不修改现有reward实现。

## 判据

- A：seed0/1/2在候选公式下的最高分checkpoint全部通过expert gate；
- B：仅2/3通过；
- C：少于2/3通过。

