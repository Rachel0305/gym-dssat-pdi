# 021_40 SY2014 agent/demo n-step梯度来源拆分

精确重放021_35 1K训练，只增加`n-step demo`和`n-step agent`的独立梯度范数及其与demo margin的余弦日志。训练结果必须通过五个checkpoint online-Q参数哈希与021_39逐一相同来证明没有改变轨迹；不重复DSSAT评估。

## 预注册判据

分别计算collapse(50–500)与recovery(501–1000)：

- 若`cos(agent nstep, margin)`阶段差>=0.2且大于demo对应差，agent n-step为优先候选；
- 若`cos(demo nstep, margin)`阶段差>=0.2且大于agent对应差，demo n-step为优先候选；
- 两者均>=0.2且差值<0.05：共同候选；
- 均<0.2：无明确来源。

只定位候选，不现场修改loss。

