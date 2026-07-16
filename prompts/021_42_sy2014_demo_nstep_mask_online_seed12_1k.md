# 021_42 SY2014 demo n-step屏蔽在线seed1/2 1K复核

## 目的

固定同一个021_34 seed0离线优质起点与021_41唯一干预，只改变在线action RNG和mixed-sampling RNG，验证1K保持性是否跨在线随机轨迹复现。

## 设置

- seed1：action_seed=1，sample_seed=21036；
- seed2：action_seed=2，sample_seed=21037；
- 离线500次模型、optimizer状态、环境、reward、IC、动作、预算不变；
- demo n-step在线权重=0，其他loss与021_41相同；
- checkpoint 250/500/750/1000；不启动5K；
- 本实验不是不同网络初始化seed，必须明确标注为online stochasticity复核。

## 判据

每个seed成功：至少3/4 checkpoint满足yield>=11077、I<=120、N<=300、DAP>90 N=0，且1000步通过、最低产量>=9613。

- A：seed1和seed2均成功；
- B：仅一个成功；
- C：均失败。

