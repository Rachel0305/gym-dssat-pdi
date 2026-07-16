# 021_38 SY2014 demonstration n-step 与 margin 梯度冲突审计

## 目的与边界

在021_35的0/250/500/750/1000 checkpoint上，使用同一组160条冻结standardized oracle demonstration，分别计算group-balanced demonstration n-step loss与large-margin loss对online Q网络的梯度范数及余弦夹角。

- 纯离线，不训练、不optimizer.step、不调用DSSAT；
- no-op与nonzero两组各占0.5期望权重，近似021_35 demo半批8/8的期望；
- agent transition未保存，本任务不评估agent梯度；
- 不根据结果改权重。

## 预注册判据

- conflict checkpoint：demo n-step/margin梯度范数比>=10且余弦<=-0.3；
- A：250和500均为conflict，而0或1000至少一个不是；支持“坍缩期demo梯度冲突增强”的候选；
- B：五个checkpoint全都conflict或全都不conflict；该指标缺少时间区分力；
- C：其他混合结果。

任何结果都只作机制定位，不写成因果结论。

