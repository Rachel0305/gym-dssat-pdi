# 022_02 SY2014 阶段型 Monte-Carlo-target DQN seed1短验证

## 1. 前提

022_01在独立来源的官方固定DAP动作空间中找到28/48个主科学成功方案、18/48个严格成功方案，证明目标上界存在。本任务首次检验DQN-style Q网络能否在该阶段环境中学习。

## 2. 单一结构假设

只检验以下组合：

- 每季6个官方阶段决策；
- 阶段之间底层DSSAT逐日no-op推进；
- 一季结束后，为6个transition生成terminal-complete return；
- 该return替代1-step/5-step TD，是唯一Q目标；
- 不使用demo、DQfD、margin loss、PER、target bootstrap或行为克隆。

本任务不是对未修改SB3 DQN的测试，方法名为：`stage-based DQN-style Q network with terminal-complete Monte Carlo targets`。

## 3. 冻结环境

- SY2014、IC=2、PDI/DSSAT 4.8.0；
- 阶段DAP：1/30/50/65/85/110；
- 动作9档：I∈{0,15,30} × N∈{0,50,100}；
- I≤120、N≤300；
- DAP110仅允许N=0动作0/1/2；
- 阶段间非决策日强制no-op；
- 固定使用021_24已经验证的25维观测标准化器，不增加信息、不重新计算统计量。

## 4. 阶段环境验证门槛

训练前必须用动作序列 `[3,4,7,1,1,0]` 回放022_01的 `W60_critical + N200_early`：

- 阶段DAP严格为1/30/50/65/85/110；
- HWAM=11202±2；
- I=60、N=200；
- return首项的原始值=`11202-5408+1620-60-5×200=6354`；
- 底层非阶段日全部执行no-op；
- DAP110含N动作必须报错而不是裁剪混叠。

任一失败则停止，不训练。

## 5. Q目标与尺度

```text
raw G_t = sum(k=t..T) r_k
training target = raw G_t / 1000
loss = SmoothL1(Q(s_t,a_t), raw G_t/1000)
```

除以1000是统一线性单位缩放：产量增益、I/N成本和feasibility bonus全部同时缩放，不改变任何候选的排序。固定值在训练前写死，不做扫描。

## 6. 预注册训练参数

- seed=1；
- 网络：25→64→64→9，ReLU；
- optimizer=Adam，learning_rate=1e-4；
- replay=uniform，capacity=2000 stage transitions；
- batch_size=32；
- learning starts=5完整季节；
- 每完成一季，执行6次gradient update；
- gradient clip=10；
- 共60个训练季，即最多360个stage interactions；
- epsilon按季从1.0线性降到0.2；
- checkpoint=15/30/45/60季；
- 每个checkpoint只做一次deterministic评估；
- 不自动延长训练，不扫描学习率、epsilon或return scale。

## 7. 预注册结果判定

每个checkpoint按022_01主科学判据与严格附加等级评估。

- **A 初步成功**：至少3/4 checkpoint通过主科学判据，episode60通过，且至少一个checkpoint通过严格等级；允许补一个独立seed。
- **B 有信号但不稳定**：至少一个checkpoint通过主科学判据，但不满足A；停止自动扩展，先报告轨迹。
- **C 失败**：没有checkpoint通过主科学判据；停止本结构，不现场调参。
- **D 实现失败**：阶段回放、动作、总量、Summary或数值有限性校验失败；停止科学解释。

单个漂亮尖峰不能判成功。

## 8. 输出

- 阶段环境回放验证JSON/CSV；
- 训练季轨迹CSV；
- update loss/gradient/Q日志；
- 4个checkpoint模型；
- checkpoint确定性评估summary和阶段动作CSV；
- PNG/SVG；
- 中文实验记录。

