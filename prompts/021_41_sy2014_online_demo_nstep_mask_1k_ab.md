# 021_41 SY2014 在线demo n-step屏蔽1K A/B

## 假设

021_40显示demo n-step与demo margin在坍缩、恢复两阶段均强烈冲突（余弦约-0.705），agent n-step与margin小幅同向。测试在线阶段删除demo n-step梯度能否保持021_34优质起点。

## 单变量

- Control：复用021_35，不重跑；
- Treatment：与021_35完全相同，仅将在线`TDn_all`改为“只保留agent样本的n-step项”；
- agent n-step每条样本对总loss的缩放保持原来`sum/32`，不是重新按16条求均值；
- demo margin、agent TD1、L2、采样、探索、target、reward、IC、动作和预算均不变；
- 离线500次event-balanced学习不变；
- seed0、1K、checkpoint 250/500/750/1000；不自动5K。

## 判据

checkpoint通过条件同021_35：yield>=11077、I<=120、N<=300、DAP>90 N=0。

- A：Treatment至少3/4通过、1000步通过、最低产量>=9613，且通过数高于Control的1/4；
- B：Treatment通过数高于Control但不满足A，或最终通过但过程仍不稳；
- C：Treatment通过数<=Control，未改善在线保持性。

不根据结果调整其他权重。

