# 021_11 SY2014 seed1 中期下降与25K恢复机制离线审计

## 目标

解释021_10延长探索配置下seed1的轨迹：10K为N300/11046 kg/ha，15K降至N100/9620，20K降至N50/7352，25K恢复N300/10794。

本轮只做离线诊断，判断下降与恢复是否伴随：

1. online/target Q氮动作排序改变；
2. target网络更新；
3. replay中新写入经验的氮请求比例变化；
4. 固定buffer上的Bellman/TD残差改变。

## 输入

- 021_10 seed1 10K、15K、20K、25K checkpoint和replay buffer；
- 021_06已保存的18个SY2014真实固定观测，用于同状态Q比较；
- 9动作映射。

## 边界

- 不训练、不调用DSSAT；
- 不修改reward、IC、探索、target interval、动作、预算或观测；
- 固定观测来自seed0参考轨迹，只用于网络同状态比较，不冒充seed1实际轨迹；
- replay保存的是请求动作，不冒充wrapper实际执行动作；
- 事后TD残差不称为历史训练loss；
- 不根据结果直接启动seed2或修改环境。

## 方法

1. 对10K、15K、20K、25K读取`_n_calls`、epsilon及online/target参数哈希；
2. 将同一组18个固定观测输入四个checkpoint网络，输出9动作online/target Q、全局argmax及每个灌溉档位N0/N50/N100完整排序；
3. 比较10K→15K、15K→20K、20K→25K排序变化；
4. 从endpoint环形replay buffer精确提取各窗口新增transition，统计请求N0/N50/N100、正灌溉、联合动作、动作熵和reward分布；
5. 对每个窗口endpoint buffer分别用窗口起点和终点网络计算Bellman残差，排除buffer变化混杂；
6. 输出CSV、JSON、PNG和中文实验记录。

## 判定纪律

- target更新与下降/恢复时间重合不等于因果；
- 若20K→25K恢复同时出现高氮请求回升和online排序回升，只能支持“策略与经验分布共同恢复”；
- 若fixed-buffer残差改善，可说明数值拟合改善，但不能单独确定超参数根因；
- 观测空间部分可观测问题继续作为独立设计决策，不在本任务修改。
