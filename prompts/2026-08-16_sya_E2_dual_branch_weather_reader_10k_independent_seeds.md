# E2 双通道天气读取独立 seed 10K 验证

## 目的

seed0 在5K保留天气响应和多样动作，但10K出现天气响应、动作多样性和产量同时坍缩。本任务分别运行seed1和seed2到10K，保存2K/5K/10K，判断5K表现及5K到10K的变化是否具有跨seed稳定性。

## 固定条件

除随机seed外，天气、双通道结构、reward、安全mask、16动作格、PPO参数、训练/验证年份和DSSAT输入均与144E2一致。seed1与seed2使用独立输出目录和模型名，串行运行，不覆盖seed0。

## 边界与判定

- 每个seed总步数10000，checkpoint=[2000,5000,10000]；禁止25K以上；
- 分别报告5K/10K的产量、灌溉、N、PFP_N、动作序列和非零动作组合；
- 对每个seed的5K与10K执行相同的同状态天气交换，>1%才视为保留天气响应；
- 若多个seed均表现为5K较好而10K坍缩，说明checkpoint时长效应具有一定跨seed重复性；
- 若seed之间方向不一致，则不能冻结5K为稳定checkpoint，应报告高度seed依赖；
- 本轮只验证E2自身跨seed稳定性，不额外训练no-forecast seed1/seed2；因此no-forecast优势仍只在seed0配对成立。

